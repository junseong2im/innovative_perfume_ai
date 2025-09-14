from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from decimal import Decimal
import asyncio
from collections import defaultdict

from ..core.logging_config import get_logger, performance_logger
from ..core.exceptions import SystemException, ValidationException, ErrorCode
from ..database.connection import get_db_session

logger = get_logger(__name__)


class PartnershipType(str, Enum):
    """제휴 유형"""
    BRAND_COLLABORATION = "brand_collaboration"     # 브랜드 협업
    INGREDIENT_SUPPLIER = "ingredient_supplier"     # 원료 공급업체
    MANUFACTURING = "manufacturing"                 # 제조 파트너
    RETAIL_PARTNER = "retail_partner"              # 판매 파트너
    MARKETING_AFFILIATE = "marketing_affiliate"     # 마케팅 제휴


class RevenueModel(str, Enum):
    """수익 모델"""
    COMMISSION = "commission"           # 커미션 (판매액의 %)
    FIXED_FEE = "fixed_fee"            # 고정 수수료
    REVENUE_SHARE = "revenue_share"     # 매출 분배
    LICENSING = "licensing"             # 라이선스 수수료
    SUBSCRIPTION = "subscription"       # 구독 수수료
    PER_RECIPE = "per_recipe"          # 레시피당 수수료


class PartnerStatus(str, Enum):
    """파트너 상태"""
    PENDING = "pending"                 # 승인 대기
    ACTIVE = "active"                  # 활성
    SUSPENDED = "suspended"            # 일시 중단
    TERMINATED = "terminated"          # 종료


@dataclass
class Partner:
    """파트너 정보"""
    partner_id: str
    company_name: str
    partnership_type: PartnershipType
    status: PartnerStatus
    
    # 기본 정보
    contact_info: Dict[str, str]
    business_info: Dict[str, Any]
    brand_info: Optional[Dict[str, Any]] = None
    
    # 계약 조건
    revenue_model: RevenueModel
    commission_rate: Optional[Decimal] = None
    fixed_fee_amount: Optional[Decimal] = None
    minimum_guarantee: Optional[Decimal] = None
    
    # 제품/서비스 정보
    available_products: List[Dict[str, Any]]
    service_capabilities: List[str]
    quality_standards: Dict[str, Any]
    
    # 성과 지표
    total_revenue: Decimal = Decimal('0.00')
    total_orders: int = 0
    avg_rating: float = 0.0
    customer_satisfaction: float = 0.0
    
    # 계약 정보
    contract_start_date: datetime
    contract_end_date: Optional[datetime] = None
    auto_renewal: bool = True
    
    created_at: datetime
    updated_at: datetime


@dataclass
class Transaction:
    """거래 정보"""
    transaction_id: str
    partner_id: str
    user_id: str
    recipe_id: Optional[str]
    
    # 거래 상세
    transaction_type: str  # purchase, commission, licensing
    amount: Decimal
    commission_amount: Optional[Decimal] = None
    
    # 제품 정보
    products: List[Dict[str, Any]]
    total_quantity: int = 1
    
    # 상태
    status: str  # pending, completed, cancelled, refunded
    payment_status: str  # pending, paid, failed, refunded
    
    # 메타데이터
    metadata: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None


class PartnershipService:
    """파트너십 서비스"""
    
    def __init__(self):
        self.partners: Dict[str, Partner] = {}
        self.transactions: List[Transaction] = []
        
        # 수익 추적
        self.revenue_tracking = {
            "daily": defaultdict(Decimal),
            "monthly": defaultdict(Decimal),
            "partner": defaultdict(Decimal)
        }
        
        # 초기 파트너 데이터 로드
        self._initialize_demo_partners()
    
    def _initialize_demo_partners(self):
        """데모 파트너 초기화"""
        
        # 프리미엄 브랜드 파트너들
        demo_partners = [
            {
                "company_name": "크리드 (CREED)",
                "partnership_type": PartnershipType.BRAND_COLLABORATION,
                "contact_info": {
                    "email": "partnership@creed.com",
                    "phone": "+33-1-2345-6789",
                    "address": "Paris, France"
                },
                "business_info": {
                    "founded": 1760,
                    "headquarters": "Paris, France",
                    "annual_revenue": "€200M+",
                    "employee_count": "500+",
                    "market_position": "luxury"
                },
                "brand_info": {
                    "brand_tier": "ultra_luxury",
                    "signature_notes": ["iris", "sandalwood", "bergamot", "rose"],
                    "price_range": "$200-$500",
                    "target_demographic": "luxury_consumers"
                },
                "revenue_model": RevenueModel.COMMISSION,
                "commission_rate": Decimal('0.15'),  # 15%
                "available_products": [
                    {
                        "category": "eau_de_parfum",
                        "sizes": ["30ml", "50ml", "100ml"],
                        "customization_available": True,
                        "min_order_quantity": 10
                    },
                    {
                        "category": "discovery_set",
                        "sizes": ["5ml x 8"],
                        "customization_available": False,
                        "min_order_quantity": 1
                    }
                ],
                "service_capabilities": [
                    "custom_blending",
                    "luxury_packaging",
                    "personalized_engraving",
                    "vip_consultation"
                ],
                "quality_standards": {
                    "ingredient_purity": "99%+",
                    "longevity": "8+ hours",
                    "sillage": "moderate_to_heavy",
                    "certifications": ["IFRA", "ISO"]
                }
            },
            {
                "company_name": "메종 마르지엘라 (Maison Margiela)",
                "partnership_type": PartnershipType.BRAND_COLLABORATION,
                "contact_info": {
                    "email": "business@maisonmargiela.com",
                    "phone": "+33-1-3456-7890",
                    "address": "Paris, France"
                },
                "business_info": {
                    "founded": 1988,
                    "headquarters": "Paris, France",
                    "annual_revenue": "€500M+",
                    "employee_count": "1000+",
                    "market_position": "contemporary_luxury"
                },
                "brand_info": {
                    "brand_tier": "contemporary_luxury",
                    "signature_notes": ["cashmeran", "white_musk", "cedar", "vanilla"],
                    "price_range": "$80-$200",
                    "target_demographic": "contemporary_luxury_consumers"
                },
                "revenue_model": RevenueModel.REVENUE_SHARE,
                "commission_rate": Decimal('0.12'),  # 12%
                "available_products": [
                    {
                        "category": "replica_line",
                        "sizes": ["30ml", "100ml"],
                        "customization_available": True,
                        "min_order_quantity": 5
                    },
                    {
                        "category": "custom_blend",
                        "sizes": ["50ml"],
                        "customization_available": True,
                        "min_order_quantity": 1
                    }
                ],
                "service_capabilities": [
                    "memory_scent_recreation",
                    "artistic_packaging",
                    "concept_development",
                    "limited_editions"
                ],
                "quality_standards": {
                    "ingredient_purity": "95%+",
                    "longevity": "6+ hours",
                    "sillage": "moderate",
                    "certifications": ["IFRA"]
                }
            },
            {
                "company_name": "원료 전문 - 지보당 (Givaudan)",
                "partnership_type": PartnershipType.INGREDIENT_SUPPLIER,
                "contact_info": {
                    "email": "fragrance@givaudan.com",
                    "phone": "+41-22-780-9111",
                    "address": "Vernier, Switzerland"
                },
                "business_info": {
                    "founded": 1895,
                    "headquarters": "Vernier, Switzerland",
                    "annual_revenue": "CHF 6.2B",
                    "employee_count": "15,000+",
                    "market_position": "market_leader"
                },
                "revenue_model": RevenueModel.PER_RECIPE,
                "fixed_fee_amount": Decimal('25.00'),  # 레시피당 $25
                "available_products": [
                    {
                        "category": "natural_ingredients",
                        "count": "2000+",
                        "origin": "global",
                        "sustainability": "certified"
                    },
                    {
                        "category": "synthetic_molecules",
                        "count": "3000+",
                        "innovation": "proprietary",
                        "safety": "RIFM_approved"
                    }
                ],
                "service_capabilities": [
                    "ingredient_sourcing",
                    "quality_assurance",
                    "regulatory_compliance",
                    "sustainability_consulting",
                    "innovation_partnership"
                ],
                "quality_standards": {
                    "purity_standards": "pharmaceutical_grade",
                    "sustainability": "100% traceable",
                    "certifications": ["ISO", "RSPO", "Organic"],
                    "testing": "comprehensive"
                }
            }
        ]
        
        for partner_data in demo_partners:
            partner_id = str(uuid.uuid4())
            partner = Partner(
                partner_id=partner_id,
                status=PartnerStatus.ACTIVE,
                total_revenue=Decimal('0.00'),
                total_orders=0,
                avg_rating=4.5,
                customer_satisfaction=0.85,
                contract_start_date=datetime.utcnow(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                **partner_data
            )
            self.partners[partner_id] = partner
            
            logger.info(f"Demo partner initialized: {partner.company_name}")
    
    async def register_partner(self, partner_data: Dict[str, Any]) -> str:
        """파트너 등록"""
        
        try:
            partner_id = str(uuid.uuid4())
            
            # 데이터 검증
            self._validate_partner_data(partner_data)
            
            partner = Partner(
                partner_id=partner_id,
                status=PartnerStatus.PENDING,
                total_revenue=Decimal('0.00'),
                total_orders=0,
                avg_rating=0.0,
                customer_satisfaction=0.0,
                contract_start_date=datetime.utcnow(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                **partner_data
            )
            
            self.partners[partner_id] = partner
            
            # 승인 워크플로우 시작
            await self._initiate_approval_workflow(partner)
            
            logger.info(f"Partner registered: {partner.company_name} ({partner_id})")
            
            return partner_id
            
        except Exception as e:
            logger.error(f"Partner registration failed: {e}")
            raise SystemException(
                message=f"파트너 등록 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _validate_partner_data(self, partner_data: Dict[str, Any]):
        """파트너 데이터 검증"""
        
        required_fields = [
            "company_name", "partnership_type", "contact_info",
            "business_info", "revenue_model"
        ]
        
        for field in required_fields:
            if field not in partner_data:
                raise ValidationException(f"필수 필드 누락: {field}")
        
        # 연락처 정보 검증
        contact_info = partner_data.get("contact_info", {})
        if not contact_info.get("email") or "@" not in contact_info["email"]:
            raise ValidationException("유효한 이메일 주소가 필요합니다")
        
        # 수익 모델 검증
        revenue_model = partner_data["revenue_model"]
        if revenue_model == RevenueModel.COMMISSION and not partner_data.get("commission_rate"):
            raise ValidationException("커미션 모델에는 커미션율이 필요합니다")
        
        if revenue_model == RevenueModel.FIXED_FEE and not partner_data.get("fixed_fee_amount"):
            raise ValidationException("고정 수수료 모델에는 금액이 필요합니다")
    
    async def _initiate_approval_workflow(self, partner: Partner):
        """승인 워크플로우 시작"""
        
        try:
            # 자동 검증 단계
            validation_score = await self._calculate_partner_score(partner)
            
            if validation_score >= 0.8:
                # 자동 승인
                partner.status = PartnerStatus.ACTIVE
                await self._send_approval_notification(partner, auto_approved=True)
            elif validation_score >= 0.6:
                # 수동 리뷰 필요
                await self._queue_manual_review(partner, validation_score)
            else:
                # 자동 거절
                partner.status = PartnerStatus.TERMINATED
                await self._send_rejection_notification(partner, validation_score)
            
        except Exception as e:
            logger.error(f"Approval workflow failed for {partner.partner_id}: {e}")
    
    async def _calculate_partner_score(self, partner: Partner) -> float:
        """파트너 점수 계산"""
        
        score = 0.0
        
        # 비즈니스 정보 점수 (0.4)
        business_info = partner.business_info
        if business_info.get("annual_revenue"):
            score += 0.2
        if business_info.get("employee_count"):
            score += 0.1
        if business_info.get("market_position") in ["luxury", "market_leader"]:
            score += 0.1
        
        # 연락처 완성도 점수 (0.2)
        contact_info = partner.contact_info
        if contact_info.get("email") and "@" in contact_info["email"]:
            score += 0.1
        if contact_info.get("phone"):
            score += 0.05
        if contact_info.get("address"):
            score += 0.05
        
        # 제품/서비스 역량 점수 (0.3)
        if partner.available_products:
            score += 0.15
        if partner.service_capabilities:
            score += 0.1
        if partner.quality_standards:
            score += 0.05
        
        # 수익 모델 적절성 점수 (0.1)
        if partner.revenue_model in [RevenueModel.COMMISSION, RevenueModel.REVENUE_SHARE]:
            score += 0.1
        
        return min(1.0, score)
    
    async def _send_approval_notification(self, partner: Partner, auto_approved: bool = False):
        """승인 알림 발송"""
        
        notification = {
            "partner_id": partner.partner_id,
            "company_name": partner.company_name,
            "approval_type": "automatic" if auto_approved else "manual",
            "message": "축하합니다! 파트너십이 승인되었습니다.",
            "next_steps": [
                "API 키 발급 및 설정",
                "제품 카탈로그 업로드",
                "첫 번째 테스트 주문 처리",
                "성과 대시보드 설정"
            ]
        }
        
        # 실제로는 이메일/SMS 발송
        logger.info(f"Approval notification sent to {partner.company_name}")
    
    async def generate_partner_recommendations(
        self,
        user_profile: Dict[str, Any],
        recipe_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """사용자 맞춤 파트너 추천"""
        
        try:
            recommendations = []
            
            # 사용자 프로필 분석
            budget = user_profile.get("budget", 100)
            quality_preference = user_profile.get("quality_preference", "medium")
            brand_preference = user_profile.get("brand_preference", [])
            
            # 레시피 분석
            complexity = recipe_data.get("complexity", 5)
            price_category = self._estimate_price_category(recipe_data, complexity)
            
            # 파트너 필터링 및 점수 계산
            for partner in self.partners.values():
                if partner.status != PartnerStatus.ACTIVE:
                    continue
                
                compatibility_score = self._calculate_compatibility_score(
                    partner, user_profile, recipe_data
                )
                
                if compatibility_score > 0.5:  # 임계값
                    recommendations.append({
                        "partner_id": partner.partner_id,
                        "company_name": partner.company_name,
                        "partnership_type": partner.partnership_type.value,
                        "compatibility_score": compatibility_score,
                        "estimated_price": self._estimate_price(partner, recipe_data),
                        "estimated_delivery": self._estimate_delivery_time(partner),
                        "quality_rating": partner.avg_rating,
                        "special_offers": self._get_special_offers(partner, user_profile),
                        "why_recommended": self._generate_recommendation_reason(
                            partner, compatibility_score, user_profile
                        )
                    })
            
            # 점수순 정렬
            recommendations.sort(key=lambda x: x["compatibility_score"], reverse=True)
            
            return recommendations[:5]  # 상위 5개
            
        except Exception as e:
            logger.error(f"Partner recommendation generation failed: {e}")
            raise SystemException(
                message=f"파트너 추천 생성 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _calculate_compatibility_score(
        self,
        partner: Partner,
        user_profile: Dict[str, Any],
        recipe_data: Dict[str, Any]
    ) -> float:
        """파트너 호환성 점수 계산"""
        
        score = 0.0
        
        # 예산 호환성 (0.3)
        budget = user_profile.get("budget", 100)
        estimated_price = self._estimate_price(partner, recipe_data)
        
        if estimated_price <= budget:
            score += 0.3
        elif estimated_price <= budget * 1.2:  # 20% 초과 허용
            score += 0.2
        elif estimated_price <= budget * 1.5:  # 50% 초과
            score += 0.1
        
        # 품질 선호도 (0.25)
        quality_pref = user_profile.get("quality_preference", "medium")
        brand_tier = partner.brand_info.get("brand_tier", "standard") if partner.brand_info else "standard"
        
        quality_match = {
            ("high", "ultra_luxury"): 0.25,
            ("high", "luxury"): 0.20,
            ("medium", "contemporary_luxury"): 0.25,
            ("medium", "luxury"): 0.20,
            ("low", "standard"): 0.25,
        }
        
        score += quality_match.get((quality_pref, brand_tier), 0.1)
        
        # 브랜드 선호도 (0.2)
        brand_prefs = user_profile.get("brand_preference", [])
        if partner.company_name in brand_prefs:
            score += 0.2
        elif any(brand.lower() in partner.company_name.lower() for brand in brand_prefs):
            score += 0.1
        
        # 서비스 역량 (0.15)
        required_services = recipe_data.get("required_services", [])
        available_services = partner.service_capabilities
        
        if required_services:
            match_ratio = len(set(required_services) & set(available_services)) / len(required_services)
            score += 0.15 * match_ratio
        else:
            score += 0.15  # 특별 요구사항 없음
        
        # 과거 성과 (0.1)
        if partner.avg_rating >= 4.5:
            score += 0.1
        elif partner.avg_rating >= 4.0:
            score += 0.08
        elif partner.avg_rating >= 3.5:
            score += 0.05
        
        return min(1.0, score)
    
    def _estimate_price(self, partner: Partner, recipe_data: Dict[str, Any]) -> float:
        """가격 추정"""
        
        base_price = 50.0  # 기본 가격
        
        # 브랜드 티어 기준 가격 조정
        if partner.brand_info:
            brand_tier = partner.brand_info.get("brand_tier", "standard")
            tier_multipliers = {
                "ultra_luxury": 5.0,
                "luxury": 3.0,
                "contemporary_luxury": 2.0,
                "premium": 1.5,
                "standard": 1.0
            }
            base_price *= tier_multipliers.get(brand_tier, 1.0)
        
        # 복잡도 기준 조정
        complexity = recipe_data.get("complexity", 5)
        base_price *= (1.0 + (complexity - 5) * 0.1)
        
        # 특수 서비스 추가 비용
        special_services = recipe_data.get("special_services", [])
        service_costs = {
            "custom_blending": 25.0,
            "personalized_engraving": 15.0,
            "luxury_packaging": 20.0,
            "vip_consultation": 50.0
        }
        
        for service in special_services:
            base_price += service_costs.get(service, 0)
        
        return round(base_price, 2)
    
    def _estimate_delivery_time(self, partner: Partner) -> str:
        """배송 시간 추정"""
        
        if partner.partnership_type == PartnershipType.BRAND_COLLABORATION:
            if partner.brand_info and partner.brand_info.get("brand_tier") == "ultra_luxury":
                return "4-6 weeks"  # 럭셔리 브랜드는 시간이 더 걸림
            else:
                return "2-3 weeks"
        elif partner.partnership_type == PartnershipType.MANUFACTURING:
            return "1-2 weeks"
        else:
            return "3-5 business days"
    
    def _get_special_offers(self, partner: Partner, user_profile: Dict[str, Any]) -> List[str]:
        """특별 제안 조회"""
        
        offers = []
        
        # 신규 고객 할인
        if user_profile.get("is_new_customer", True):
            offers.append("신규 고객 15% 할인")
        
        # 대량 주문 할인
        if user_profile.get("order_quantity", 1) >= 5:
            offers.append("5개 이상 주문 시 10% 추가 할인")
        
        # 파트너별 특별 혜택
        if partner.company_name == "크리드 (CREED)":
            offers.append("무료 개인 맞춤 컨설팅")
        elif partner.company_name == "메종 마르지엘라 (Maison Margiela)":
            offers.append("레플리카 샘플 세트 증정")
        
        # 계절별 프로모션
        current_month = datetime.now().month
        if current_month in [11, 12]:  # 연말
            offers.append("홀리데이 시즌 특가")
        elif current_month in [2, 3]:  # 봄
            offers.append("스프링 컬렉션 선공개")
        
        return offers
    
    def _generate_recommendation_reason(
        self,
        partner: Partner,
        compatibility_score: float,
        user_profile: Dict[str, Any]
    ) -> List[str]:
        """추천 이유 생성"""
        
        reasons = []
        
        if compatibility_score >= 0.9:
            reasons.append("귀하의 취향과 완벽하게 일치합니다")
        elif compatibility_score >= 0.8:
            reasons.append("높은 호환성을 보여줍니다")
        
        if partner.avg_rating >= 4.5:
            reasons.append("고객 만족도가 매우 높습니다")
        
        if partner.brand_info:
            brand_tier = partner.brand_info.get("brand_tier")
            if brand_tier == "ultra_luxury":
                reasons.append("최고급 프리미엄 브랜드입니다")
            elif brand_tier == "luxury":
                reasons.append("럭셔리 브랜드의 품질을 보장합니다")
        
        budget = user_profile.get("budget", 100)
        estimated_price = self._estimate_price(partner, {})
        
        if estimated_price <= budget * 0.8:
            reasons.append("예산 내에서 최고의 가성비를 제공합니다")
        
        if "custom_blending" in partner.service_capabilities:
            reasons.append("개인 맞춤 블렌딩 서비스를 제공합니다")
        
        return reasons
    
    def _estimate_price_category(self, recipe_data: Dict[str, Any], complexity: int) -> str:
        """가격 카테고리 추정"""
        
        if complexity >= 8:
            return "premium"
        elif complexity >= 6:
            return "mid_range"
        else:
            return "accessible"
    
    async def process_transaction(
        self,
        user_id: str,
        partner_id: str,
        order_data: Dict[str, Any]
    ) -> str:
        """거래 처리"""
        
        try:
            if partner_id not in self.partners:
                raise ValidationException("파트너를 찾을 수 없습니다")
            
            partner = self.partners[partner_id]
            
            transaction_id = str(uuid.uuid4())
            
            # 거래 금액 계산
            amount = Decimal(str(order_data["amount"]))
            
            # 커미션 계산
            commission_amount = None
            if partner.revenue_model == RevenueModel.COMMISSION:
                commission_amount = amount * partner.commission_rate
            elif partner.revenue_model == RevenueModel.FIXED_FEE:
                commission_amount = partner.fixed_fee_amount
            
            transaction = Transaction(
                transaction_id=transaction_id,
                partner_id=partner_id,
                user_id=user_id,
                recipe_id=order_data.get("recipe_id"),
                transaction_type="purchase",
                amount=amount,
                commission_amount=commission_amount,
                products=order_data.get("products", []),
                total_quantity=order_data.get("quantity", 1),
                status="pending",
                payment_status="pending",
                metadata=order_data.get("metadata", {}),
                created_at=datetime.utcnow()
            )
            
            self.transactions.append(transaction)
            
            # 파트너에게 주문 알림
            await self._notify_partner_new_order(partner, transaction)
            
            # 수익 추적 업데이트
            self._update_revenue_tracking(partner_id, commission_amount or Decimal('0'))
            
            logger.info(f"Transaction processed: {transaction_id} for partner {partner.company_name}")
            
            performance_logger.log_execution_time(
                operation="process_transaction",
                execution_time=0.0,
                success=True,
                extra_data={
                    "partner_id": partner_id,
                    "amount": float(amount),
                    "commission": float(commission_amount) if commission_amount else 0
                }
            )
            
            return transaction_id
            
        except Exception as e:
            logger.error(f"Transaction processing failed: {e}")
            raise SystemException(
                message=f"거래 처리 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    async def _notify_partner_new_order(self, partner: Partner, transaction: Transaction):
        """파트너에게 신규 주문 알림"""
        
        notification = {
            "partner_id": partner.partner_id,
            "transaction_id": transaction.transaction_id,
            "order_details": {
                "amount": float(transaction.amount),
                "products": transaction.products,
                "quantity": transaction.total_quantity,
                "commission": float(transaction.commission_amount) if transaction.commission_amount else 0
            },
            "customer_info": {
                "user_id": transaction.user_id,
                "special_requests": transaction.metadata.get("special_requests", [])
            },
            "next_steps": [
                "주문 확인 및 승인",
                "제작/배송 일정 안내",
                "품질 검증 완료 보고",
                "배송 추적 정보 업데이트"
            ]
        }
        
        # 실제로는 파트너 시스템에 API 호출 또는 이메일 발송
        logger.info(f"Order notification sent to {partner.company_name}")
    
    def _update_revenue_tracking(self, partner_id: str, commission_amount: Decimal):
        """수익 추적 업데이트"""
        
        today = datetime.now().date().isoformat()
        month = datetime.now().strftime("%Y-%m")
        
        self.revenue_tracking["daily"][today] += commission_amount
        self.revenue_tracking["monthly"][month] += commission_amount
        self.revenue_tracking["partner"][partner_id] += commission_amount
        
        # 파트너 통계 업데이트
        if partner_id in self.partners:
            self.partners[partner_id].total_revenue += commission_amount
            self.partners[partner_id].total_orders += 1
    
    async def get_partner_analytics(
        self,
        partner_id: str,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """파트너 분석 데이터"""
        
        try:
            if partner_id not in self.partners:
                raise ValidationException("파트너를 찾을 수 없습니다")
            
            partner = self.partners[partner_id]
            
            # 지정 기간 거래 필터링
            cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
            partner_transactions = [
                t for t in self.transactions
                if t.partner_id == partner_id and t.created_at >= cutoff_date
            ]
            
            # 분석 데이터 계산
            analytics = {
                "partner_info": {
                    "company_name": partner.company_name,
                    "partnership_type": partner.partnership_type.value,
                    "status": partner.status.value,
                    "contract_duration_days": (datetime.utcnow() - partner.contract_start_date).days
                },
                "performance_metrics": {
                    "total_revenue": float(partner.total_revenue),
                    "total_orders": partner.total_orders,
                    "avg_order_value": float(partner.total_revenue / max(partner.total_orders, 1)),
                    "customer_rating": partner.avg_rating,
                    "satisfaction_rate": partner.customer_satisfaction
                },
                "recent_activity": {
                    "orders_last_30_days": len(partner_transactions),
                    "revenue_last_30_days": float(sum(t.amount for t in partner_transactions)),
                    "avg_processing_time": "2.5 days",  # 실제 계산 필요
                    "completion_rate": 0.95  # 실제 계산 필요
                },
                "trends": self._calculate_trends(partner_transactions),
                "top_products": self._get_top_products(partner_transactions),
                "recommendations": self._generate_partner_recommendations(partner, partner_transactions)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Partner analytics generation failed: {e}")
            raise SystemException(
                message=f"파트너 분석 데이터 생성 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _calculate_trends(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """트렌드 분석"""
        
        if len(transactions) < 2:
            return {"status": "insufficient_data"}
        
        # 일별 매출 트렌드
        daily_revenue = defaultdict(Decimal)
        for transaction in transactions:
            date_key = transaction.created_at.date().isoformat()
            daily_revenue[date_key] += transaction.amount
        
        # 성장률 계산 (간단한 버전)
        revenue_values = list(daily_revenue.values())
        if len(revenue_values) >= 2:
            recent_avg = float(sum(revenue_values[-7:]) / min(7, len(revenue_values)))  # 최근 7일 평균
            previous_avg = float(sum(revenue_values[:-7]) / max(1, len(revenue_values) - 7))  # 이전 기간 평균
            
            growth_rate = ((recent_avg - previous_avg) / max(previous_avg, 1)) * 100 if previous_avg else 0
        else:
            growth_rate = 0
        
        return {
            "revenue_growth_rate": round(growth_rate, 2),
            "trend_direction": "up" if growth_rate > 5 else "down" if growth_rate < -5 else "stable",
            "daily_revenue": {date: float(amount) for date, amount in daily_revenue.items()}
        }
    
    def _get_top_products(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """인기 제품 분석"""
        
        product_stats = defaultdict(lambda: {"count": 0, "revenue": Decimal('0')})
        
        for transaction in transactions:
            for product in transaction.products:
                product_name = product.get("name", "Unknown")
                product_stats[product_name]["count"] += 1
                product_stats[product_name]["revenue"] += transaction.amount / len(transaction.products)
        
        # 상위 5개 제품
        top_products = sorted(
            product_stats.items(),
            key=lambda x: x[1]["revenue"],
            reverse=True
        )[:5]
        
        return [
            {
                "product_name": name,
                "order_count": stats["count"],
                "total_revenue": float(stats["revenue"])
            }
            for name, stats in top_products
        ]
    
    def _generate_partner_recommendations(
        self,
        partner: Partner,
        transactions: List[Transaction]
    ) -> List[str]:
        """파트너 개선 권장사항"""
        
        recommendations = []
        
        # 주문량 기반 권장사항
        if len(transactions) < 5:
            recommendations.append("마케팅 활동을 통해 주문량을 늘려보세요")
        
        # 평점 기반 권장사항
        if partner.avg_rating < 4.0:
            recommendations.append("고객 서비스 품질 개선이 필요합니다")
        elif partner.avg_rating < 4.5:
            recommendations.append("일관된 품질 관리로 평점을 더 높일 수 있습니다")
        
        # 수익 모델 최적화
        if partner.revenue_model == RevenueModel.COMMISSION and partner.commission_rate < Decimal('0.10'):
            recommendations.append("커미션율 조정을 검토해보세요")
        
        # 서비스 확장 제안
        if len(partner.service_capabilities) < 3:
            recommendations.append("추가 서비스 제공으로 경쟁력을 높여보세요")
        
        return recommendations
    
    async def get_revenue_dashboard(self) -> Dict[str, Any]:
        """수익 대시보드 데이터"""
        
        try:
            today = datetime.now().date().isoformat()
            month = datetime.now().strftime("%Y-%m")
            
            dashboard = {
                "summary": {
                    "total_partners": len(self.partners),
                    "active_partners": len([p for p in self.partners.values() if p.status == PartnerStatus.ACTIVE]),
                    "total_transactions": len(self.transactions),
                    "total_revenue": float(sum(self.revenue_tracking["partner"].values()))
                },
                "today_stats": {
                    "revenue": float(self.revenue_tracking["daily"].get(today, Decimal('0'))),
                    "transactions": len([t for t in self.transactions if t.created_at.date().isoformat() == today])
                },
                "monthly_stats": {
                    "revenue": float(self.revenue_tracking["monthly"].get(month, Decimal('0'))),
                    "target": 50000.0,  # 월 목표 수익
                    "achievement_rate": float(self.revenue_tracking["monthly"].get(month, Decimal('0'))) / 50000.0
                },
                "top_partners": self._get_top_partners(),
                "revenue_trends": self._get_revenue_trends(),
                "partnership_distribution": self._get_partnership_distribution()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Revenue dashboard generation failed: {e}")
            raise SystemException(
                message=f"수익 대시보드 생성 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _get_top_partners(self) -> List[Dict[str, Any]]:
        """상위 파트너 조회"""
        
        active_partners = [p for p in self.partners.values() if p.status == PartnerStatus.ACTIVE]
        top_partners = sorted(active_partners, key=lambda p: p.total_revenue, reverse=True)[:5]
        
        return [
            {
                "company_name": partner.company_name,
                "partnership_type": partner.partnership_type.value,
                "total_revenue": float(partner.total_revenue),
                "total_orders": partner.total_orders,
                "avg_rating": partner.avg_rating
            }
            for partner in top_partners
        ]
    
    def _get_revenue_trends(self) -> Dict[str, List[float]]:
        """수익 트렌드 데이터"""
        
        # 최근 30일 일별 수익
        daily_revenues = []
        for i in range(30, 0, -1):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            daily_revenues.append(float(self.revenue_tracking["daily"].get(date, Decimal('0'))))
        
        # 최근 12개월 월별 수익
        monthly_revenues = []
        for i in range(12, 0, -1):
            date = (datetime.now() - timedelta(days=i*30)).strftime("%Y-%m")
            monthly_revenues.append(float(self.revenue_tracking["monthly"].get(date, Decimal('0'))))
        
        return {
            "daily": daily_revenues,
            "monthly": monthly_revenues
        }
    
    def _get_partnership_distribution(self) -> Dict[str, int]:
        """파트너십 유형 분포"""
        
        distribution = defaultdict(int)
        for partner in self.partners.values():
            distribution[partner.partnership_type.value] += 1
        
        return dict(distribution)


# 전역 파트너십 서비스 인스턴스
partnership_service = PartnershipService()