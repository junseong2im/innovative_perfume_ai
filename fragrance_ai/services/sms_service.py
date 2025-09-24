"""
SMS 서비스 관리자
- 다중 SMS 제공업체 지원
- 템플릿 기반 SMS
- 대량 SMS 처리
- 전송 상태 추적
"""

import asyncio
import aiohttp
import hashlib
import hmac
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import uuid
import re

from ..core.logging_config import get_logger
from ..core.monitoring import MetricsCollector

logger = get_logger(__name__)

class SMSProvider(Enum):
    TWILIO = "twilio"
    AWS_SNS = "aws_sns"
    ALIGO = "aligo"  # 한국 SMS 서비스
    COOLSMS = "coolsms"  # 한국 SMS 서비스
    NAVER_SENS = "naver_sens"  # 네이버 클라우드 플랫폼

class SMSStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNDELIVERED = "undelivered"

class SMSType(Enum):
    SMS = "sms"  # 90자 이하
    LMS = "lms"  # 2000자 이하
    MMS = "mms"  # 멀티미디어

@dataclass
class SMSRecipient:
    phone_number: str
    name: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SMSTemplate:
    name: str
    content: str
    sms_type: SMSType = SMSType.SMS
    variables: List[str] = field(default_factory=list)

@dataclass
class SMSMessage:
    id: str
    recipients: List[SMSRecipient]
    content: str
    sender_number: str
    template_name: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    sms_type: SMSType = SMSType.SMS
    scheduled_at: Optional[datetime] = None
    status: SMSStatus = SMSStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    provider_message_id: Optional[str] = None
    cost: Optional[float] = None

class SMSService:
    """SMS 서비스 관리자"""

    def __init__(
        self,
        provider: SMSProvider = SMSProvider.COOLSMS,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sender_number: Optional[str] = None,
        # Twilio 설정
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        # AWS SNS 설정
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_region: str = "ap-northeast-2",
        # 네이버 클라우드 설정
        naver_access_key: Optional[str] = None,
        naver_secret_key: Optional[str] = None,
        naver_service_id: Optional[str] = None,
        # 일반 설정
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        self.provider = provider
        self.api_key = api_key
        self.api_secret = api_secret
        self.sender_number = sender_number
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 제공업체별 설정
        self.provider_config = {
            "twilio": {
                "account_sid": twilio_account_sid,
                "auth_token": twilio_auth_token,
                "base_url": "https://api.twilio.com/2010-04-01"
            },
            "aws_sns": {
                "access_key": aws_access_key,
                "secret_key": aws_secret_key,
                "region": aws_region
            },
            "coolsms": {
                "api_key": api_key,
                "api_secret": api_secret,
                "base_url": "https://api.coolsms.co.kr/sms/4"
            },
            "aligo": {
                "api_key": api_key,
                "user_id": api_secret,  # Aligo는 user_id 사용
                "base_url": "https://apis.aligo.in"
            },
            "naver_sens": {
                "access_key": naver_access_key,
                "secret_key": naver_secret_key,
                "service_id": naver_service_id,
                "base_url": f"https://sens.apigw.ntruss.com/sms/v2/services/{naver_service_id}"
            }
        }

        # 템플릿 관리
        self.templates: Dict[str, SMSTemplate] = {}

        # SMS 큐 및 상태 추적
        self.sms_queue: List[SMSMessage] = []
        self.sms_status: Dict[str, SMSStatus] = {}
        self.delivery_stats = {
            "sent": 0,
            "delivered": 0,
            "failed": 0,
            "undelivered": 0,
            "total_cost": 0.0
        }

        # HTTP 클라이언트
        self.session: Optional[aiohttp.ClientSession] = None

        # 백그라운드 작업
        self._processor_task: Optional[asyncio.Task] = None
        self._status_checker_task: Optional[asyncio.Task] = None

        # 메트릭
        self.metrics_collector = MetricsCollector()

    async def initialize(self):
        """SMS 서비스 초기화"""
        try:
            logger.info(f"Initializing SMS service with provider: {self.provider.value}")

            # HTTP 세션 설정
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # 기본 템플릿 로드
            await self._load_default_templates()

            # 제공업체별 검증
            await self._validate_provider_config()

            # 백그라운드 프로세서 시작
            self._processor_task = asyncio.create_task(self._sms_processor())
            self._status_checker_task = asyncio.create_task(self._status_checker())

            logger.info("SMS service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SMS service: {e}")
            raise

    async def _load_default_templates(self):
        """기본 SMS 템플릿 로드"""
        default_templates = {
            "verification": SMSTemplate(
                name="verification",
                content="[{{app_name}}] 인증번호: {{code}} ({{expiry_minutes}}분 유효)",
                sms_type=SMSType.SMS,
                variables=["app_name", "code", "expiry_minutes"]
            ),
            "welcome": SMSTemplate(
                name="welcome",
                content="[{{app_name}}] {{name}}님, 회원가입을 환영합니다! 향수 AI와 함께 특별한 향수를 찾아보세요.",
                sms_type=SMSType.LMS,
                variables=["app_name", "name"]
            ),
            "notification": SMSTemplate(
                name="notification",
                content="[{{app_name}}] {{message}}",
                sms_type=SMSType.SMS,
                variables=["app_name", "message"]
            ),
            "marketing": SMSTemplate(
                name="marketing",
                content="[{{app_name}}] 🌸 {{name}}님을 위한 특별한 향수 추천이 도착했습니다! {{url}} (수신거부: {{unsubscribe_url}})",
                sms_type=SMSType.LMS,
                variables=["app_name", "name", "url", "unsubscribe_url"]
            )
        }

        self.templates.update(default_templates)
        logger.info(f"Loaded {len(default_templates)} default SMS templates")

    async def _validate_provider_config(self):
        """제공업체 설정 검증"""
        config = self.provider_config[self.provider.value]

        if self.provider == SMSProvider.TWILIO:
            if not config.get("account_sid") or not config.get("auth_token"):
                raise ValueError("Twilio account_sid and auth_token are required")

        elif self.provider == SMSProvider.COOLSMS:
            if not config.get("api_key") or not config.get("api_secret"):
                raise ValueError("CoolSMS API key and secret are required")

        elif self.provider == SMSProvider.NAVER_SENS:
            if not all([config.get("access_key"), config.get("secret_key"), config.get("service_id")]):
                raise ValueError("Naver Cloud Platform credentials are required")

        # 간단한 연결 테스트 (선택적)
        await self._test_connection()

    async def _test_connection(self):
        """API 연결 테스트"""
        try:
            if self.provider == SMSProvider.COOLSMS:
                # CoolSMS 잔액 조회로 연결 테스트
                await self._coolsms_get_balance()

            logger.info(f"{self.provider.value} connection test successful")

        except Exception as e:
            logger.warning(f"Connection test failed (non-critical): {e}")

    async def send_sms(
        self,
        recipients: Union[str, List[str], List[SMSRecipient]],
        content: str,
        sms_type: SMSType = SMSType.SMS,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """SMS 전송"""

        # 수신자 처리
        if isinstance(recipients, str):
            recipients = [SMSRecipient(phone_number=self._normalize_phone(recipients))]
        elif isinstance(recipients, list) and recipients and isinstance(recipients[0], str):
            recipients = [SMSRecipient(phone_number=self._normalize_phone(phone)) for phone in recipients]

        # SMS 메시지 생성
        sms_id = str(uuid.uuid4())
        sms_message = SMSMessage(
            id=sms_id,
            recipients=recipients,
            content=content,
            sender_number=self.sender_number,
            sms_type=self._determine_sms_type(content) if sms_type == SMSType.SMS else sms_type,
            scheduled_at=scheduled_at
        )

        # 큐에 추가
        self.sms_queue.append(sms_message)
        self.sms_status[sms_id] = SMSStatus.PENDING

        logger.info(f"SMS queued: {sms_id}")
        return sms_id

    async def send_template_sms(
        self,
        recipients: Union[str, List[str], List[SMSRecipient]],
        template_name: str,
        template_variables: Dict[str, Any]
    ) -> str:
        """템플릿 SMS 전송"""

        if template_name not in self.templates:
            raise ValueError(f"SMS template not found: {template_name}")

        template = self.templates[template_name]

        # 템플릿 렌더링
        content = template.content
        for var, value in template_variables.items():
            content = content.replace(f"{{{{{var}}}}}", str(value))

        return await self.send_sms(
            recipients=recipients,
            content=content,
            sms_type=template.sms_type
        )

    async def send_verification_sms(
        self,
        phone_number: str,
        code: str,
        app_name: str = "Fragrance AI",
        expiry_minutes: int = 5
    ) -> str:
        """인증 SMS 전송"""

        return await self.send_template_sms(
            recipients=phone_number,
            template_name="verification",
            template_variables={
                "app_name": app_name,
                "code": code,
                "expiry_minutes": expiry_minutes
            }
        )

    async def _sms_processor(self):
        """SMS 처리 백그라운드 작업"""
        while True:
            try:
                if not self.sms_queue:
                    await asyncio.sleep(5)
                    continue

                # 스케줄된 SMS만 처리
                current_time = datetime.now(timezone.utc)
                sms_to_process = [
                    sms for sms in self.sms_queue
                    if sms.scheduled_at is None or sms.scheduled_at <= current_time
                ]

                for sms in sms_to_process[:5]:  # 한 번에 최대 5개 처리
                    try:
                        await self._send_single_sms(sms)
                        self.sms_queue.remove(sms)

                    except Exception as e:
                        logger.error(f"Failed to send SMS {sms.id}: {e}")
                        sms.error_message = str(e)
                        sms.status = SMSStatus.FAILED
                        self.sms_status[sms.id] = SMSStatus.FAILED
                        self.sms_queue.remove(sms)

                await asyncio.sleep(2)  # 2초 대기 (rate limiting)

            except Exception as e:
                logger.error(f"SMS processor error: {e}")
                await asyncio.sleep(10)

    async def _send_single_sms(self, sms_message: SMSMessage):
        """단일 SMS 전송"""
        try:
            if self.provider == SMSProvider.COOLSMS:
                await self._send_via_coolsms(sms_message)
            elif self.provider == SMSProvider.TWILIO:
                await self._send_via_twilio(sms_message)
            elif self.provider == SMSProvider.NAVER_SENS:
                await self._send_via_naver_sens(sms_message)
            elif self.provider == SMSProvider.ALIGO:
                await self._send_via_aligo(sms_message)
            elif self.provider == SMSProvider.TOAST_SMS:
                await self._send_via_toast_sms(sms_message)
            else:
                # 기본 HTTP SMS 게이트웨이 사용
                await self._send_via_generic_gateway(sms_message)

            # 상태 업데이트
            sms_message.status = SMSStatus.SENT
            sms_message.sent_at = datetime.now(timezone.utc)
            self.sms_status[sms_message.id] = SMSStatus.SENT
            self.delivery_stats["sent"] += 1

            logger.info(f"SMS sent successfully: {sms_message.id}")

        except Exception as e:
            sms_message.status = SMSStatus.FAILED
            sms_message.error_message = str(e)
            self.sms_status[sms_message.id] = SMSStatus.FAILED
            self.delivery_stats["failed"] += 1
            raise

    async def _send_via_coolsms(self, sms_message: SMSMessage):
        """CoolSMS를 통한 SMS 전송"""
        config = self.provider_config["coolsms"]

        for recipient in sms_message.recipients:
            payload = {
                "message": {
                    "to": recipient.phone_number,
                    "from": sms_message.sender_number,
                    "text": sms_message.content,
                    "type": sms_message.sms_type.value.upper()
                }
            }

            headers = {
                "Authorization": f"HMAC-SHA256 apiKey={config['api_key']}, date={self._get_iso_time()}, salt={self._generate_salt()}, signature={self._generate_coolsms_signature(payload)}",
                "Content-Type": "application/json"
            }

            async with self.session.post(
                f"{config['base_url']}/send",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    sms_message.provider_message_id = result.get("groupId")

                    # 비용 계산 (예시)
                    sms_message.cost = 0.02 if sms_message.sms_type == SMSType.SMS else 0.05
                    self.delivery_stats["total_cost"] += sms_message.cost
                else:
                    error_text = await response.text()
                    raise Exception(f"CoolSMS API error: {response.status} - {error_text}")

    async def _send_via_twilio(self, sms_message: SMSMessage):
        """Twilio를 통한 SMS 전송"""
        config = self.provider_config["twilio"]
        auth = aiohttp.BasicAuth(config["account_sid"], config["auth_token"])

        for recipient in sms_message.recipients:
            payload = {
                "To": recipient.phone_number,
                "From": sms_message.sender_number,
                "Body": sms_message.content
            }

            async with self.session.post(
                f"{config['base_url']}/Accounts/{config['account_sid']}/Messages.json",
                data=payload,
                auth=auth
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    sms_message.provider_message_id = result.get("sid")

                    # 비용 정보 (Twilio에서 제공)
                    sms_message.cost = float(result.get("price", 0) or 0)
                    self.delivery_stats["total_cost"] += sms_message.cost
                else:
                    error_text = await response.text()
                    raise Exception(f"Twilio API error: {response.status} - {error_text}")

    async def _send_via_naver_sens(self, sms_message: SMSMessage):
        """네이버 클라우드 플랫폼 SENS를 통한 SMS 전송"""
        config = self.provider_config["naver_sens"]

        messages = []
        for recipient in sms_message.recipients:
            messages.append({
                "to": recipient.phone_number,
                "content": sms_message.content
            })

        payload = {
            "type": sms_message.sms_type.value.upper(),
            "from": sms_message.sender_number,
            "content": sms_message.content,
            "messages": messages
        }

        # NAVER API 서명 생성
        timestamp = str(int(datetime.now().timestamp() * 1000))
        signature = self._generate_naver_signature(timestamp, payload)

        headers = {
            "Content-Type": "application/json",
            "x-ncp-apigw-timestamp": timestamp,
            "x-ncp-iam-access-key": config["access_key"],
            "x-ncp-apigw-signature-v2": signature
        }

        async with self.session.post(
            f"{config['base_url']}/messages",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 202:
                result = await response.json()
                sms_message.provider_message_id = result.get("requestId")

                # 비용 계산 (예시)
                sms_message.cost = len(messages) * (0.008 if sms_message.sms_type == SMSType.SMS else 0.024)
                self.delivery_stats["total_cost"] += sms_message.cost
            else:
                error_text = await response.text()
                raise Exception(f"NAVER SENS API error: {response.status} - {error_text}")

    async def _status_checker(self):
        """전송 상태 확인 작업"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다 체크

                # 전송된 메시지들의 상태 확인
                # 실제로는 각 제공업체의 상태 조회 API를 사용

            except Exception as e:
                logger.error(f"Status checker error: {e}")

    async def _coolsms_get_balance(self):
        """CoolSMS 잔액 조회"""
        config = self.provider_config["coolsms"]

        headers = {
            "Authorization": f"HMAC-SHA256 apiKey={config['api_key']}, date={self._get_iso_time()}, salt={self._generate_salt()}, signature={self._generate_coolsms_signature({})}",
            "Content-Type": "application/json"
        }

        async with self.session.get(
            f"{config['base_url']}/balance",
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("balance", {})
            else:
                error_text = await response.text()
                raise Exception(f"CoolSMS balance API error: {response.status} - {error_text}")

    def _normalize_phone(self, phone_number: str) -> str:
        """전화번호 정규화"""
        # 숫자만 추출
        digits_only = re.sub(r'\D', '', phone_number)

        # 한국 번호 처리
        if digits_only.startswith('82'):
            return '+' + digits_only
        elif digits_only.startswith('0'):
            return '+82' + digits_only[1:]
        elif len(digits_only) == 10 or len(digits_only) == 11:
            if digits_only.startswith('10') or digits_only.startswith('11'):
                return '+82' + digits_only
            else:
                return '+82' + digits_only

        return phone_number  # 그대로 반환

    def _determine_sms_type(self, content: str) -> SMSType:
        """내용 길이에 따른 SMS 타입 결정"""
        content_length = len(content)

        if content_length <= 90:
            return SMSType.SMS
        elif content_length <= 2000:
            return SMSType.LMS
        else:
            return SMSType.MMS

    def _get_iso_time(self) -> str:
        """ISO 시간 문자열 반환"""
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    def _generate_salt(self) -> str:
        """랜덤 솔트 생성"""
        import secrets
        return secrets.token_hex(16)

    def _generate_coolsms_signature(self, payload: dict) -> str:
        """CoolSMS API 서명 생성"""
        config = self.provider_config["coolsms"]

        data = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
        salt = self._generate_salt()
        date = self._get_iso_time()

        message = date + salt + data

        signature = hmac.new(
            config["api_secret"].encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _generate_naver_signature(self, timestamp: str, payload: dict) -> str:
        """네이버 클라우드 플랫폼 API 서명 생성"""
        config = self.provider_config["naver_sens"]

        method = "POST"
        uri = f"/sms/v2/services/{config['service_id']}/messages"

        message = f"{method} {uri}\n{timestamp}\n{config['access_key']}"

        signature = hmac.new(
            config["secret_key"].encode(),
            message.encode(),
            hashlib.sha256
        ).digest()

        import base64
        return base64.b64encode(signature).decode()

    def get_sms_status(self, sms_id: str) -> Optional[SMSStatus]:
        """SMS 상태 조회"""
        return self.sms_status.get(sms_id)

    def get_delivery_stats(self) -> Dict[str, Any]:
        """전송 통계 조회"""
        total_sms = sum([
            self.delivery_stats["sent"],
            self.delivery_stats["failed"],
            self.delivery_stats["undelivered"]
        ])

        return {
            "total_sms": total_sms,
            "sent": self.delivery_stats["sent"],
            "delivered": self.delivery_stats["delivered"],
            "failed": self.delivery_stats["failed"],
            "undelivered": self.delivery_stats["undelivered"],
            "success_rate": (
                (self.delivery_stats["sent"] / total_sms * 100)
                if total_sms > 0 else 0
            ),
            "total_cost": round(self.delivery_stats["total_cost"], 4),
            "queue_size": len(self.sms_queue),
            "provider": self.provider.value
        }

    async def _send_via_aligo(self, sms_message: SMSMessage):
        """알리고 SMS를 통한 전송 (한국 서비스)"""
        config = self.provider_config.get("aligo", {})

        for recipient in sms_message.recipients:
            payload = {
                "key": config.get("api_key"),
                "user_id": config.get("user_id"),
                "sender": sms_message.sender_number,
                "receiver": recipient.phone_number,
                "msg": sms_message.content,
                "testmode_yn": "N" if config.get("production", False) else "Y"
            }

            async with self.session.post(
                "https://apis.aligo.in/send/",
                data=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("result_code") == "1":
                        sms_message.provider_message_id = result.get("msg_id")
                        logger.info(f"Aligo SMS sent: {result.get('msg_id')}")
                    else:
                        raise Exception(f"Aligo error: {result.get('message')}")
                else:
                    raise Exception(f"Aligo HTTP error: {response.status}")

    async def _send_via_toast_sms(self, sms_message: SMSMessage):
        """NHN Toast SMS를 통한 전송"""
        config = self.provider_config.get("toast_sms", {})

        headers = {
            "Content-Type": "application/json",
            "X-Secret-Key": config.get("secret_key")
        }

        for recipient in sms_message.recipients:
            payload = {
                "body": sms_message.content,
                "sendNo": sms_message.sender_number,
                "recipientList": [{
                    "recipientNo": recipient.phone_number,
                    "recipientName": recipient.name
                }],
                "userId": config.get("user_id"),
                "statsId": sms_message.id[:8]  # 통계 ID
            }

            endpoint = f"https://api-sms.cloud.toast.com/sms/v3.0/appKeys/{config.get('app_key')}/sender/sms"

            async with self.session.post(
                endpoint,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("header", {}).get("isSuccessful"):
                        sms_message.provider_message_id = result.get("body", {}).get("data", {}).get("requestId")
                        logger.info(f"Toast SMS sent: {sms_message.provider_message_id}")
                    else:
                        raise Exception(f"Toast SMS error: {result.get('header', {}).get('resultMessage')}")
                else:
                    raise Exception(f"Toast SMS HTTP error: {response.status}")

    async def _send_via_generic_gateway(self, sms_message: SMSMessage):
        """일반 HTTP SMS 게이트웨이를 통한 전송"""
        # 환경변수에서 일반 게이트웨이 설정 읽기
        gateway_url = os.environ.get("SMS_GATEWAY_URL", "http://localhost:8080/sms/send")
        gateway_token = os.environ.get("SMS_GATEWAY_TOKEN", "")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {gateway_token}"
        } if gateway_token else {"Content-Type": "application/json"}

        for recipient in sms_message.recipients:
            payload = {
                "from": sms_message.sender_number,
                "to": recipient.phone_number,
                "message": sms_message.content,
                "type": sms_message.sms_type.value,
                "callback_url": f"{os.environ.get('BASE_URL', 'http://localhost:8000')}/api/v1/sms/webhook",
                "metadata": {
                    "message_id": sms_message.id,
                    "recipient_name": recipient.name,
                    "template_code": sms_message.template_code
                }
            }

            try:
                async with self.session.post(
                    gateway_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status in [200, 201, 202]:
                        result = await response.json()
                        sms_message.provider_message_id = result.get("message_id", sms_message.id)
                        logger.info(f"Generic SMS gateway sent: {sms_message.provider_message_id}")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Generic gateway error ({response.status}): {error_text}")
            except asyncio.TimeoutError:
                raise Exception("SMS gateway timeout after 30 seconds")
            except Exception as e:
                logger.error(f"Generic gateway error: {e}")
                raise

    async def shutdown(self):
        """서비스 종료"""
        try:
            # 백그라운드 작업 중지
            if self._processor_task:
                self._processor_task.cancel()
            if self._status_checker_task:
                self._status_checker_task.cancel()

            # HTTP 세션 종료
            if self.session:
                await self.session.close()

            logger.info("SMS service shutdown completed")

        except Exception as e:
            logger.error(f"SMS service shutdown error: {e}")


# 전역 SMS 서비스 인스턴스
sms_service: Optional[SMSService] = None

def get_sms_service() -> Optional[SMSService]:
    """글로벌 SMS 서비스 반환"""
    return sms_service

async def initialize_sms_service(**kwargs) -> SMSService:
    """SMS 서비스 초기화"""
    global sms_service

    sms_service = SMSService(**kwargs)
    await sms_service.initialize()

    return sms_service