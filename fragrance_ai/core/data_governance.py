# 🛡️ 완벽한 데이터 거버넌스 및 컴플라이언스 시스템
import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import aiofiles
import re
from collections import defaultdict
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import structlog

logger = structlog.get_logger("data_governance")


class DataClassification(Enum):
    """데이터 분류"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceStandard(Enum):
    """컴플라이언스 표준"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    KOREAN_PIPA = "korean_pipa"


class DataOperation(Enum):
    """데이터 작업"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    SHARE = "share"
    ANONYMIZE = "anonymize"
    PURGE = "purge"


class ConsentType(Enum):
    """동의 유형"""
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    OPT_IN = "opt_in"
    OPT_OUT = "opt_out"


@dataclass
class DataSubject:
    """데이터 주체"""
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    nationality: Optional[str] = None
    age: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consent_status: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataAsset:
    """데이터 자산"""
    id: str
    name: str
    description: str
    classification: DataClassification
    data_type: str
    schema: Dict[str, Any]
    location: str
    owner: str
    steward: str
    retention_period: int  # days
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    compliance_requirements: List[ComplianceStandard] = field(default_factory=list)
    encryption_required: bool = True
    anonymization_required: bool = False
    tags: Set[str] = field(default_factory=set)


@dataclass
class AccessRequest:
    """접근 요청"""
    id: str
    requester_id: str
    data_asset_id: str
    operation: DataOperation
    purpose: str
    justification: str
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    status: str = "pending"
    expires_at: Optional[datetime] = None


@dataclass
class AuditRecord:
    """감사 기록"""
    id: str
    timestamp: datetime
    user_id: str
    operation: DataOperation
    data_asset_id: str
    result: str  # success, failure, blocked
    ip_address: str
    user_agent: str
    purpose: Optional[str] = None
    compliance_context: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class PrivacyRule:
    """프라이버시 규칙"""
    id: str
    name: str
    description: str
    condition: str  # SQL-like condition
    action: str  # mask, encrypt, block, log
    priority: int
    applicable_standards: List[ComplianceStandard]
    enabled: bool = True


class PIIDetector:
    """개인식별정보 탐지기"""

    def __init__(self):
        self.patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}-\d{3,4}-\d{4}\b|\b\d{11}\b'),
            "ssn": re.compile(r'\b\d{6}-\d{7}\b'),  # Korean RRN
            "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            "korean_name": re.compile(r'[가-힣]{2,4}'),
        }

        self.sensitive_fields = {
            "password", "passwd", "secret", "token", "key", "private",
            "confidential", "ssn", "social", "credit", "card", "account",
            "bank", "financial", "medical", "health", "biometric"
        }

    def scan_text(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 PII 탐지"""
        findings = {}

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                findings[pii_type] = matches

        return findings

    def scan_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """딕셔너리에서 PII 탐지"""
        findings = {}

        for key, value in data.items():
            key_lower = key.lower()

            # 필드명 기반 탐지
            for sensitive_field in self.sensitive_fields:
                if sensitive_field in key_lower:
                    findings[key] = {"type": "sensitive_field", "value": str(value)}
                    break

            # 값 기반 탐지
            if isinstance(value, str):
                text_findings = self.scan_text(value)
                if text_findings:
                    findings[key] = {"type": "content_match", "patterns": text_findings}

        return findings


class DataEncryption:
    """데이터 암호화"""

    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        self.fernet = Fernet(master_key)
        self.master_key = master_key

    def encrypt_data(self, data: Union[str, bytes, Dict]) -> str:
        """데이터 암호화"""
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False)
        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self.fernet.encrypt(data)
        return base64.b64encode(encrypted).decode('ascii')

    def decrypt_data(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')

    def generate_hash(self, data: str) -> str:
        """데이터 해시 생성"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def anonymize_data(self, data: Dict[str, Any], pii_fields: Set[str]) -> Dict[str, Any]:
        """데이터 익명화"""
        anonymized = data.copy()

        for field in pii_fields:
            if field in anonymized:
                # 해시로 대체
                original_value = str(anonymized[field])
                anonymized[field] = self.generate_hash(original_value)

        return anonymized


class ConsentManager:
    """동의 관리"""

    def __init__(self):
        self.consents: Dict[str, Dict] = {}
        self.consent_templates: Dict[str, Dict] = {}

    async def record_consent(
        self,
        subject_id: str,
        purpose: str,
        consent_type: ConsentType,
        granted: bool,
        details: Optional[Dict] = None
    ) -> str:
        """동의 기록"""
        consent_id = str(uuid.uuid4())

        consent_record = {
            "id": consent_id,
            "subject_id": subject_id,
            "purpose": purpose,
            "consent_type": consent_type.value,
            "granted": granted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
            "withdrawn": False,
            "withdrawn_at": None
        }

        if subject_id not in self.consents:
            self.consents[subject_id] = {}

        self.consents[subject_id][purpose] = consent_record

        await logger.ainfo(
            f"동의 기록됨: {subject_id} - {purpose} - {'승인' if granted else '거부'}"
        )

        return consent_id

    async def withdraw_consent(self, subject_id: str, purpose: str) -> bool:
        """동의 철회"""
        if subject_id in self.consents and purpose in self.consents[subject_id]:
            self.consents[subject_id][purpose]["withdrawn"] = True
            self.consents[subject_id][purpose]["withdrawn_at"] = datetime.now(timezone.utc).isoformat()

            await logger.ainfo(f"동의 철회됨: {subject_id} - {purpose}")
            return True

        return False

    def check_consent(self, subject_id: str, purpose: str) -> bool:
        """동의 확인"""
        if subject_id not in self.consents:
            return False

        consent = self.consents[subject_id].get(purpose)
        if not consent:
            return False

        return consent["granted"] and not consent["withdrawn"]

    def get_consent_history(self, subject_id: str) -> List[Dict]:
        """동의 이력 조회"""
        if subject_id not in self.consents:
            return []

        return list(self.consents[subject_id].values())


class DataRetentionManager:
    """데이터 보존 관리"""

    def __init__(self):
        self.retention_policies: Dict[str, Dict] = {}
        self.scheduled_deletions: List[Dict] = []

    def create_retention_policy(
        self,
        policy_id: str,
        name: str,
        retention_period_days: int,
        data_types: List[str],
        conditions: Optional[Dict] = None
    ):
        """보존 정책 생성"""
        self.retention_policies[policy_id] = {
            "id": policy_id,
            "name": name,
            "retention_period_days": retention_period_days,
            "data_types": data_types,
            "conditions": conditions or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active": True
        }

    async def evaluate_retention(self, data_asset: DataAsset) -> Dict[str, Any]:
        """보존 평가"""
        evaluation = {
            "asset_id": data_asset.id,
            "should_delete": False,
            "deletion_date": None,
            "applicable_policies": [],
            "reasons": []
        }

        for policy in self.retention_policies.values():
            if not policy["active"]:
                continue

            # 데이터 타입 확인
            if data_asset.data_type in policy["data_types"]:
                evaluation["applicable_policies"].append(policy["id"])

                # 보존 기간 확인
                retention_end = data_asset.created_at + timedelta(days=policy["retention_period_days"])

                if datetime.now(timezone.utc) > retention_end:
                    evaluation["should_delete"] = True
                    evaluation["deletion_date"] = retention_end.isoformat()
                    evaluation["reasons"].append(f"보존 기간 만료: {policy['name']}")

        return evaluation

    async def schedule_deletion(self, asset_id: str, deletion_date: datetime, reason: str):
        """삭제 예약"""
        deletion_record = {
            "id": str(uuid.uuid4()),
            "asset_id": asset_id,
            "deletion_date": deletion_date.isoformat(),
            "reason": reason,
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "status": "scheduled"
        }

        self.scheduled_deletions.append(deletion_record)

        await logger.ainfo(f"삭제 예약됨: {asset_id} - {deletion_date}")


class ComplianceEngine:
    """컴플라이언스 엔진"""

    def __init__(self):
        self.rules: Dict[str, PrivacyRule] = {}
        self.violations: List[Dict] = []
        self.compliance_checks = {
            ComplianceStandard.GDPR: self._check_gdpr_compliance,
            ComplianceStandard.CCPA: self._check_ccpa_compliance,
            ComplianceStandard.KOREAN_PIPA: self._check_pipa_compliance,
        }

    def add_rule(self, rule: PrivacyRule):
        """규칙 추가"""
        self.rules[rule.id] = rule

    async def evaluate_compliance(
        self,
        operation: DataOperation,
        data_asset: DataAsset,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """컴플라이언스 평가"""
        evaluation = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "required_actions": [],
            "risk_score": 0.0
        }

        # 각 컴플라이언스 표준 확인
        for standard in data_asset.compliance_requirements:
            if standard in self.compliance_checks:
                check_result = await self.compliance_checks[standard](
                    operation, data_asset, user_id, context
                )

                if not check_result["compliant"]:
                    evaluation["compliant"] = False
                    evaluation["violations"].extend(check_result["violations"])

                evaluation["warnings"].extend(check_result.get("warnings", []))
                evaluation["required_actions"].extend(check_result.get("required_actions", []))
                evaluation["risk_score"] = max(evaluation["risk_score"], check_result.get("risk_score", 0))

        return evaluation

    async def _check_gdpr_compliance(
        self,
        operation: DataOperation,
        data_asset: DataAsset,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GDPR 컴플라이언스 확인"""
        result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "required_actions": [],
            "risk_score": 0.0
        }

        # 개인 데이터 처리 시 동의 확인 필요
        if operation in [DataOperation.CREATE, DataOperation.READ, DataOperation.UPDATE]:
            if "consent_verified" not in context or not context["consent_verified"]:
                result["compliant"] = False
                result["violations"].append("GDPR: 개인 데이터 처리 시 동의가 필요합니다")
                result["risk_score"] = 8.0

        # 데이터 보존 기간 확인
        if data_asset.retention_period > 2555:  # 7년 초과
            result["warnings"].append("GDPR: 보존 기간이 권장 기간을 초과합니다")

        # 암호화 요구사항
        if data_asset.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            if not data_asset.encryption_required:
                result["violations"].append("GDPR: 민감한 개인 데이터는 암호화가 필요합니다")
                result["compliant"] = False
                result["risk_score"] = max(result["risk_score"], 7.0)

        return result

    async def _check_ccpa_compliance(
        self,
        operation: DataOperation,
        data_asset: DataAsset,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """CCPA 컴플라이언스 확인"""
        result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "required_actions": [],
            "risk_score": 0.0
        }

        # 개인 정보 판매 시 옵트아웃 권리 확인
        if operation == DataOperation.SHARE:
            if "opt_out_verified" not in context:
                result["warnings"].append("CCPA: 개인 정보 공유 시 옵트아웃 권리를 확인하세요")

        return result

    async def _check_pipa_compliance(
        self,
        operation: DataOperation,
        data_asset: DataAsset,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """개인정보보호법(PIPA) 컴플라이언스 확인"""
        result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "required_actions": [],
            "risk_score": 0.0
        }

        # 개인정보 수집 시 동의 확인
        if operation == DataOperation.CREATE:
            if "collection_consent" not in context:
                result["compliant"] = False
                result["violations"].append("PIPA: 개인정보 수집 시 명시적 동의가 필요합니다")
                result["risk_score"] = 9.0

        # 고유식별정보 처리 시 별도 동의
        if "sensitive_data" in context and context["sensitive_data"]:
            if "sensitive_consent" not in context:
                result["compliant"] = False
                result["violations"].append("PIPA: 고유식별정보 처리 시 별도 동의가 필요합니다")
                result["risk_score"] = max(result["risk_score"], 8.5)

        return result


class DataGovernanceManager:
    """데이터 거버넌스 관리자"""

    def __init__(self):
        self.data_assets: Dict[str, DataAsset] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        self.access_requests: Dict[str, AccessRequest] = {}
        self.audit_records: List[AuditRecord] = []

        # 컴포넌트들
        self.pii_detector = PIIDetector()
        self.encryption = DataEncryption()
        self.consent_manager = ConsentManager()
        self.retention_manager = DataRetentionManager()
        self.compliance_engine = ComplianceEngine()

        # 설정
        self.auto_encrypt_pii = True
        self.auto_audit = True
        self.risk_threshold = 7.0

        logger.info("데이터 거버넌스 시스템 초기화 완료")

    async def register_data_asset(self, asset: DataAsset) -> str:
        """데이터 자산 등록"""
        # PII 스캔
        if isinstance(asset.schema, dict):
            pii_findings = self.pii_detector.scan_dict(asset.schema)
            if pii_findings:
                await logger.awarning(f"PII 탐지됨 in {asset.name}: {list(pii_findings.keys())}")

                # 자동 암호화 설정
                if self.auto_encrypt_pii:
                    asset.encryption_required = True
                    asset.classification = max(asset.classification, DataClassification.CONFIDENTIAL)

        self.data_assets[asset.id] = asset

        await logger.ainfo(f"데이터 자산 등록됨: {asset.name} ({asset.classification.value})")
        return asset.id

    async def request_data_access(
        self,
        requester_id: str,
        data_asset_id: str,
        operation: DataOperation,
        purpose: str,
        justification: str
    ) -> str:
        """데이터 접근 요청"""
        request_id = str(uuid.uuid4())

        access_request = AccessRequest(
            id=request_id,
            requester_id=requester_id,
            data_asset_id=data_asset_id,
            operation=operation,
            purpose=purpose,
            justification=justification
        )

        self.access_requests[request_id] = access_request

        # 자동 승인 조건 확인
        if await self._should_auto_approve(access_request):
            await self.approve_access_request(request_id, "system")

        await logger.ainfo(f"데이터 접근 요청됨: {requester_id} -> {data_asset_id}")
        return request_id

    async def _should_auto_approve(self, request: AccessRequest) -> bool:
        """자동 승인 조건 확인"""
        if request.data_asset_id not in self.data_assets:
            return False

        asset = self.data_assets[request.data_asset_id]

        # 공개 데이터는 자동 승인
        if asset.classification == DataClassification.PUBLIC:
            return True

        # 읽기 전용이고 내부 데이터인 경우 자동 승인
        if request.operation == DataOperation.READ and asset.classification == DataClassification.INTERNAL:
            return True

        return False

    async def approve_access_request(self, request_id: str, approver_id: str) -> bool:
        """접근 요청 승인"""
        if request_id not in self.access_requests:
            return False

        request = self.access_requests[request_id]
        request.approved_by = approver_id
        request.approved_at = datetime.now(timezone.utc)
        request.status = "approved"
        request.expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

        await logger.ainfo(f"데이터 접근 승인됨: {request_id} by {approver_id}")
        return True

    async def execute_data_operation(
        self,
        user_id: str,
        operation: DataOperation,
        data_asset_id: str,
        context: Dict[str, Any],
        ip_address: str = "unknown",
        user_agent: str = "unknown"
    ) -> Dict[str, Any]:
        """데이터 작업 실행"""
        result = {
            "success": False,
            "message": "",
            "data": None,
            "compliance_report": None
        }

        # 데이터 자산 확인
        if data_asset_id not in self.data_assets:
            result["message"] = "데이터 자산을 찾을 수 없습니다"
            return result

        asset = self.data_assets[data_asset_id]

        # 컴플라이언스 평가
        compliance_report = await self.compliance_engine.evaluate_compliance(
            operation, asset, user_id, context
        )

        result["compliance_report"] = compliance_report

        # 위험 점수 확인
        if compliance_report["risk_score"] > self.risk_threshold:
            result["message"] = f"위험 점수가 임계값을 초과했습니다: {compliance_report['risk_score']}"
            await self._record_audit(
                user_id, operation, data_asset_id, "blocked",
                ip_address, user_agent, context.get("purpose"),
                compliance_report, compliance_report["risk_score"]
            )
            return result

        # 컴플라이언스 위반 확인
        if not compliance_report["compliant"]:
            result["message"] = f"컴플라이언스 위반: {compliance_report['violations']}"
            await self._record_audit(
                user_id, operation, data_asset_id, "blocked",
                ip_address, user_agent, context.get("purpose"),
                compliance_report, compliance_report["risk_score"]
            )
            return result

        # 작업 실행 (실제 구현에서는 데이터베이스 작업 등)
        try:
            # 시뮬레이션된 작업 실행
            result["success"] = True
            result["message"] = "작업이 성공적으로 실행되었습니다"
            result["data"] = {"operation": operation.value, "asset": asset.name}

            # 성공 감사 기록
            await self._record_audit(
                user_id, operation, data_asset_id, "success",
                ip_address, user_agent, context.get("purpose"),
                compliance_report, compliance_report["risk_score"]
            )

        except Exception as e:
            result["message"] = f"작업 실행 중 오류: {str(e)}"
            await self._record_audit(
                user_id, operation, data_asset_id, "failure",
                ip_address, user_agent, context.get("purpose"),
                compliance_report, compliance_report["risk_score"]
            )

        return result

    async def _record_audit(
        self,
        user_id: str,
        operation: DataOperation,
        data_asset_id: str,
        result: str,
        ip_address: str,
        user_agent: str,
        purpose: Optional[str],
        compliance_context: Dict[str, Any],
        risk_score: float
    ):
        """감사 기록"""
        if not self.auto_audit:
            return

        audit_record = AuditRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            operation=operation,
            data_asset_id=data_asset_id,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            purpose=purpose,
            compliance_context=compliance_context,
            risk_score=risk_score
        )

        self.audit_records.append(audit_record)

        # 고위험 작업은 별도 로깅
        if risk_score > self.risk_threshold:
            await logger.aerror(
                f"고위험 데이터 작업: {operation.value} on {data_asset_id} by {user_id}",
                risk_score=risk_score,
                result=result
            )

    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """컴플라이언스 대시보드"""
        total_assets = len(self.data_assets)
        encrypted_assets = sum(1 for asset in self.data_assets.values() if asset.encryption_required)
        high_risk_assets = sum(
            1 for asset in self.data_assets.values()
            if asset.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]
        )

        recent_audits = [audit for audit in self.audit_records if
                        audit.timestamp > datetime.now(timezone.utc) - timedelta(days=7)]

        violations = [audit for audit in recent_audits if audit.result == "blocked"]
        high_risk_operations = [audit for audit in recent_audits if audit.risk_score > self.risk_threshold]

        return {
            "summary": {
                "total_data_assets": total_assets,
                "encrypted_assets": encrypted_assets,
                "encryption_rate": (encrypted_assets / total_assets * 100) if total_assets > 0 else 0,
                "high_risk_assets": high_risk_assets,
                "total_audit_records": len(self.audit_records)
            },
            "recent_activity": {
                "operations_last_7_days": len(recent_audits),
                "violations_last_7_days": len(violations),
                "high_risk_operations": len(high_risk_operations),
                "average_risk_score": sum(audit.risk_score for audit in recent_audits) / len(recent_audits) if recent_audits else 0
            },
            "compliance_status": {
                "gdpr_compliant_assets": self._count_compliant_assets(ComplianceStandard.GDPR),
                "ccpa_compliant_assets": self._count_compliant_assets(ComplianceStandard.CCPA),
                "pipa_compliant_assets": self._count_compliant_assets(ComplianceStandard.KOREAN_PIPA),
            },
            "top_violations": self._get_top_violations(),
            "retention_alerts": await self._get_retention_alerts()
        }

    def _count_compliant_assets(self, standard: ComplianceStandard) -> int:
        """특정 표준 준수 자산 수"""
        return sum(
            1 for asset in self.data_assets.values()
            if standard in asset.compliance_requirements
        )

    def _get_top_violations(self) -> List[Dict[str, Any]]:
        """주요 위반 사항"""
        violation_counts = defaultdict(int)

        for audit in self.audit_records:
            if audit.result == "blocked":
                for violation in audit.compliance_context.get("violations", []):
                    violation_counts[violation] += 1

        return [
            {"violation": violation, "count": count}
            for violation, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    async def _get_retention_alerts(self) -> List[Dict[str, Any]]:
        """보존 기간 경고"""
        alerts = []

        for asset in self.data_assets.values():
            evaluation = await self.retention_manager.evaluate_retention(asset)
            if evaluation["should_delete"]:
                alerts.append({
                    "asset_id": asset.id,
                    "asset_name": asset.name,
                    "deletion_date": evaluation["deletion_date"],
                    "reasons": evaluation["reasons"]
                })

        return alerts

    async def export_audit_report(self, start_date: datetime, end_date: datetime) -> str:
        """감사 보고서 내보내기"""
        filtered_audits = [
            audit for audit in self.audit_records
            if start_date <= audit.timestamp <= end_date
        ]

        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_operations": len(filtered_audits),
                "successful_operations": len([a for a in filtered_audits if a.result == "success"]),
                "failed_operations": len([a for a in filtered_audits if a.result == "failure"]),
                "blocked_operations": len([a for a in filtered_audits if a.result == "blocked"])
            },
            "audit_records": [asdict(audit) for audit in filtered_audits]
        }

        # JSON 파일로 저장
        report_file = f"audit_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        async with aiofiles.open(report_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))

        await logger.ainfo(f"감사 보고서 생성됨: {report_file}")
        return report_file

    async def cleanup_expired_data(self):
        """만료된 데이터 정리"""
        cleaned_count = 0

        for asset in list(self.data_assets.values()):
            evaluation = await self.retention_manager.evaluate_retention(asset)
            if evaluation["should_delete"]:
                # 실제 구현에서는 데이터베이스에서 삭제
                await logger.ainfo(f"만료된 데이터 자산 삭제: {asset.name}")
                del self.data_assets[asset.id]
                cleaned_count += 1

        await logger.ainfo(f"데이터 정리 완료: {cleaned_count}개 자산 삭제")
        return cleaned_count


# 전역 데이터 거버넌스 인스턴스
global_governance: Optional[DataGovernanceManager] = None


def get_governance_manager() -> DataGovernanceManager:
    """전역 거버넌스 관리자 가져오기"""
    global global_governance
    if global_governance is None:
        global_governance = DataGovernanceManager()
    return global_governance


# 편의 함수들
async def ensure_compliance(
    operation: DataOperation,
    data_asset_id: str,
    user_id: str,
    context: Dict[str, Any] = None
):
    """컴플라이언스 보장 데코레이터"""
    if context is None:
        context = {}

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            governance = get_governance_manager()

            # 컴플라이언스 검사
            result = await governance.execute_data_operation(
                user_id, operation, data_asset_id, context
            )

            if not result["success"]:
                raise PermissionError(f"컴플라이언스 위반: {result['message']}")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_consent(purpose: str, consent_type: ConsentType = ConsentType.EXPLICIT):
    """동의 필요 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(user_id: str, *args, **kwargs):
            governance = get_governance_manager()

            if not governance.consent_manager.check_consent(user_id, purpose):
                raise PermissionError(f"필요한 동의가 없습니다: {purpose}")

            return await func(user_id, *args, **kwargs)

        return wrapper
    return decorator