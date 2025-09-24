"""
이메일 서비스 관리자
- SMTP 이메일 전송
- 템플릿 기반 이메일
- 대량 이메일 처리
- 전송 상태 추적
"""

import asyncio
import smtplib
import ssl
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from jinja2 import Environment, FileSystemLoader, Template
import os
from pathlib import Path
import json
import uuid

from ..core.logging_config import get_logger
from ..core.monitoring import MetricsCollector

logger = get_logger(__name__)

class EmailProvider(Enum):
    SMTP = "smtp"
    SENDGRID = "sendgrid"
    MAILGUN = "mailgun"
    SES = "ses"

class EmailStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    FAILED = "failed"
    OPENED = "opened"
    CLICKED = "clicked"

class EmailPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class EmailRecipient:
    email: str
    name: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmailAttachment:
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"

@dataclass
class EmailTemplate:
    name: str
    subject: str
    html_content: str
    text_content: Optional[str] = None
    variables: List[str] = field(default_factory=list)

@dataclass
class EmailMessage:
    id: str
    recipients: List[EmailRecipient]
    subject: str
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    sender_email: str = ""
    sender_name: Optional[str] = None
    reply_to: Optional[str] = None
    attachments: List[EmailAttachment] = field(default_factory=list)
    template_name: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    priority: EmailPriority = EmailPriority.NORMAL
    scheduled_at: Optional[datetime] = None
    status: EmailStatus = EmailStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    tracking_enabled: bool = True

class EmailService:
    """이메일 서비스 관리자"""

    def __init__(
        self,
        provider: EmailProvider = EmailProvider.SMTP,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_use_tls: bool = True,
        templates_dir: str = "templates/email",
        default_sender: Optional[str] = None,
        enable_tracking: bool = True,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        self.provider = provider
        self.smtp_config = {
            "host": smtp_host,
            "port": smtp_port,
            "username": smtp_username,
            "password": smtp_password,
            "use_tls": smtp_use_tls
        }

        self.default_sender = default_sender
        self.enable_tracking = enable_tracking
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 템플릿 설정
        self.templates_dir = Path(templates_dir)
        self.jinja_env: Optional[Environment] = None
        self.templates: Dict[str, EmailTemplate] = {}

        # 이메일 큐 및 상태 추적
        self.email_queue: List[EmailMessage] = []
        self.email_status: Dict[str, EmailStatus] = {}
        self.delivery_stats = {
            "sent": 0,
            "delivered": 0,
            "bounced": 0,
            "failed": 0,
            "opened": 0,
            "clicked": 0
        }

        # 백그라운드 작업
        self._processor_task: Optional[asyncio.Task] = None
        self._retry_task: Optional[asyncio.Task] = None

        # 메트릭
        self.metrics_collector = MetricsCollector()

    async def initialize(self):
        """이메일 서비스 초기화"""
        try:
            logger.info("Initializing email service...")

            # 템플릿 시스템 초기화
            await self._setup_templates()

            # SMTP 연결 테스트
            if self.provider == EmailProvider.SMTP:
                await self._test_smtp_connection()

            # 백그라운드 프로세서 시작
            self._processor_task = asyncio.create_task(self._email_processor())
            self._retry_task = asyncio.create_task(self._retry_processor())

            logger.info("Email service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize email service: {e}")
            raise

    async def _setup_templates(self):
        """이메일 템플릿 설정"""
        try:
            if self.templates_dir.exists():
                self.jinja_env = Environment(
                    loader=FileSystemLoader(str(self.templates_dir)),
                    autoescape=True,
                    trim_blocks=True,
                    lstrip_blocks=True
                )

                # 기본 템플릿 로드
                await self._load_default_templates()

                logger.info(f"Email templates loaded from {self.templates_dir}")
            else:
                logger.warning(f"Templates directory not found: {self.templates_dir}")
                # 기본 템플릿만 사용
                await self._create_default_templates()

        except Exception as e:
            logger.error(f"Failed to setup email templates: {e}")
            await self._create_default_templates()

    async def _load_default_templates(self):
        """기본 템플릿 로드"""
        template_files = [
            "welcome.html",
            "verification.html",
            "password_reset.html",
            "notification.html"
        ]

        for template_file in template_files:
            template_path = self.templates_dir / template_file
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                template_name = template_path.stem
                self.templates[template_name] = EmailTemplate(
                    name=template_name,
                    subject="{{subject}}",
                    html_content=content
                )

    async def _create_default_templates(self):
        """기본 템플릿 생성"""
        default_templates = {
            "welcome": EmailTemplate(
                name="welcome",
                subject="{{app_name}}에 오신 것을 환영합니다!",
                html_content="""
                <html>
                <body>
                    <h2>{{app_name}}에 오신 것을 환영합니다, {{user_name}}님!</h2>
                    <p>저희 서비스에 가입해 주셔서 감사합니다.</p>
                    <p>향수 AI의 놀라운 기능들을 탐험해보세요.</p>
                    <a href="{{dashboard_url}}" style="background-color: #4CAF50; color: white; padding: 14px 20px; text-decoration: none;">
                        시작하기
                    </a>
                </body>
                </html>
                """,
                text_content="{{app_name}}에 오신 것을 환영합니다, {{user_name}}님!\n\n저희 서비스에 가입해 주셔서 감사합니다."
            ),

            "verification": EmailTemplate(
                name="verification",
                subject="이메일 주소를 인증해주세요",
                html_content="""
                <html>
                <body>
                    <h2>이메일 인증</h2>
                    <p>{{user_name}}님, 계정을 활성화하려면 이메일 주소를 인증해주세요.</p>
                    <p>아래 버튼을 클릭하거나 링크를 복사하여 브라우저에 붙여넣어주세요.</p>
                    <a href="{{verification_url}}" style="background-color: #2196F3; color: white; padding: 14px 20px; text-decoration: none;">
                        이메일 인증하기
                    </a>
                    <p>링크: {{verification_url}}</p>
                    <p>이 링크는 {{expiry_hours}}시간 후에 만료됩니다.</p>
                </body>
                </html>
                """,
                text_content="이메일 인증\n\n{{user_name}}님, 다음 링크로 이메일을 인증해주세요: {{verification_url}}"
            ),

            "password_reset": EmailTemplate(
                name="password_reset",
                subject="비밀번호 재설정",
                html_content="""
                <html>
                <body>
                    <h2>비밀번호 재설정</h2>
                    <p>{{user_name}}님, 비밀번호 재설정 요청을 받았습니다.</p>
                    <p>아래 버튼을 클릭하여 새로운 비밀번호를 설정해주세요.</p>
                    <a href="{{reset_url}}" style="background-color: #FF5722; color: white; padding: 14px 20px; text-decoration: none;">
                        비밀번호 재설정하기
                    </a>
                    <p>링크: {{reset_url}}</p>
                    <p>이 링크는 {{expiry_hours}}시간 후에 만료됩니다.</p>
                    <p>본인이 요청하지 않았다면 이 메일을 무시해주세요.</p>
                </body>
                </html>
                """,
                text_content="비밀번호 재설정\n\n{{user_name}}님, 다음 링크로 비밀번호를 재설정해주세요: {{reset_url}}"
            ),

            "notification": EmailTemplate(
                name="notification",
                subject="{{title}}",
                html_content="""
                <html>
                <body>
                    <h2>{{title}}</h2>
                    <p>{{message}}</p>
                    {% if action_url %}
                    <a href="{{action_url}}" style="background-color: #4CAF50; color: white; padding: 14px 20px; text-decoration: none;">
                        {{action_text or "자세히 보기"}}
                    </a>
                    {% endif %}
                </body>
                </html>
                """,
                text_content="{{title}}\n\n{{message}}"
            )
        }

        self.templates.update(default_templates)

    async def _test_smtp_connection(self):
        """SMTP 연결 테스트"""
        try:
            async with aiosmtplib.SMTP(
                hostname=self.smtp_config["host"],
                port=self.smtp_config["port"],
                use_tls=self.smtp_config["use_tls"]
            ) as server:
                if self.smtp_config["username"]:
                    await server.login(
                        self.smtp_config["username"],
                        self.smtp_config["password"]
                    )

            logger.info("SMTP connection test successful")

        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            raise

    async def send_email(
        self,
        recipients: Union[str, List[str], List[EmailRecipient]],
        subject: str,
        html_content: Optional[str] = None,
        text_content: Optional[str] = None,
        template_name: Optional[str] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[EmailAttachment]] = None,
        priority: EmailPriority = EmailPriority.NORMAL,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """이메일 전송"""

        # 수신자 처리
        if isinstance(recipients, str):
            recipients = [EmailRecipient(email=recipients)]
        elif isinstance(recipients, list) and recipients and isinstance(recipients[0], str):
            recipients = [EmailRecipient(email=email) for email in recipients]

        # 이메일 메시지 생성
        email_id = str(uuid.uuid4())
        email_message = EmailMessage(
            id=email_id,
            recipients=recipients,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            sender_email=self.default_sender,
            attachments=attachments or [],
            template_name=template_name,
            template_variables=template_variables or {},
            priority=priority,
            scheduled_at=scheduled_at
        )

        # 큐에 추가
        self.email_queue.append(email_message)
        self.email_status[email_id] = EmailStatus.PENDING

        logger.info(f"Email queued: {email_id}")
        return email_id

    async def send_template_email(
        self,
        recipients: Union[str, List[str], List[EmailRecipient]],
        template_name: str,
        template_variables: Dict[str, Any],
        priority: EmailPriority = EmailPriority.NORMAL
    ) -> str:
        """템플릿 이메일 전송"""

        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")

        return await self.send_email(
            recipients=recipients,
            subject="",  # 템플릿에서 설정
            template_name=template_name,
            template_variables=template_variables,
            priority=priority
        )

    async def send_verification_email(
        self,
        user_email: str,
        user_name: str,
        verification_url: str,
        expiry_hours: int = 24
    ) -> str:
        """이메일 인증 메일 전송"""

        return await self.send_template_email(
            recipients=user_email,
            template_name="verification",
            template_variables={
                "user_name": user_name,
                "verification_url": verification_url,
                "expiry_hours": expiry_hours
            },
            priority=EmailPriority.HIGH
        )

    async def send_welcome_email(
        self,
        user_email: str,
        user_name: str,
        app_name: str = "Fragrance AI",
        dashboard_url: str = ""
    ) -> str:
        """환영 이메일 전송"""

        return await self.send_template_email(
            recipients=user_email,
            template_name="welcome",
            template_variables={
                "user_name": user_name,
                "app_name": app_name,
                "dashboard_url": dashboard_url
            }
        )

    async def send_password_reset_email(
        self,
        user_email: str,
        user_name: str,
        reset_url: str,
        expiry_hours: int = 1
    ) -> str:
        """비밀번호 재설정 이메일 전송"""

        return await self.send_template_email(
            recipients=user_email,
            template_name="password_reset",
            template_variables={
                "user_name": user_name,
                "reset_url": reset_url,
                "expiry_hours": expiry_hours
            },
            priority=EmailPriority.HIGH
        )

    async def _email_processor(self):
        """이메일 처리 백그라운드 작업"""
        while True:
            try:
                if not self.email_queue:
                    await asyncio.sleep(5)
                    continue

                # 우선순위 순으로 정렬
                self.email_queue.sort(key=lambda x: x.priority.value, reverse=True)

                # 스케줄된 이메일만 처리
                current_time = datetime.now(timezone.utc)
                emails_to_process = [
                    email for email in self.email_queue
                    if email.scheduled_at is None or email.scheduled_at <= current_time
                ]

                for email in emails_to_process[:10]:  # 한 번에 최대 10개 처리
                    try:
                        await self._send_single_email(email)
                        self.email_queue.remove(email)

                    except Exception as e:
                        logger.error(f"Failed to send email {email.id}: {e}")
                        email.error_message = str(e)
                        email.status = EmailStatus.FAILED
                        self.email_status[email.id] = EmailStatus.FAILED

                        # 재시도 큐로 이동 (나중에 구현)
                        self.email_queue.remove(email)

                await asyncio.sleep(1)  # 1초 대기

            except Exception as e:
                logger.error(f"Email processor error: {e}")
                await asyncio.sleep(10)

    async def _send_single_email(self, email_message: EmailMessage):
        """단일 이메일 전송"""
        try:
            # 템플릿 렌더링
            if email_message.template_name:
                await self._render_template(email_message)

            # SMTP 전송
            if self.provider == EmailProvider.SMTP:
                await self._send_via_smtp(email_message)
            elif self.provider == EmailProvider.SENDGRID:
                await self._send_via_sendgrid(email_message)
            elif self.provider == EmailProvider.AWS_SES:
                await self._send_via_aws_ses(email_message)
            elif self.provider == EmailProvider.MAILGUN:
                await self._send_via_mailgun(email_message)
            elif self.provider == EmailProvider.POSTMARK:
                await self._send_via_postmark(email_message)
            elif self.provider == EmailProvider.MAILJET:
                await self._send_via_mailjet(email_message)
            else:
                # 기본 HTTP 이메일 게이트웨이 사용
                await self._send_via_generic_gateway(email_message)

            # 상태 업데이트
            email_message.status = EmailStatus.SENT
            email_message.sent_at = datetime.now(timezone.utc)
            self.email_status[email_message.id] = EmailStatus.SENT
            self.delivery_stats["sent"] += 1

            logger.info(f"Email sent successfully: {email_message.id}")

        except Exception as e:
            email_message.status = EmailStatus.FAILED
            email_message.error_message = str(e)
            self.email_status[email_message.id] = EmailStatus.FAILED
            self.delivery_stats["failed"] += 1
            raise

    async def _render_template(self, email_message: EmailMessage):
        """템플릿 렌더링"""
        if email_message.template_name not in self.templates:
            raise ValueError(f"Template not found: {email_message.template_name}")

        template = self.templates[email_message.template_name]
        variables = email_message.template_variables

        try:
            # Jinja2 템플릿 렌더링
            if self.jinja_env:
                subject_template = Template(template.subject)
                email_message.subject = subject_template.render(**variables)

                html_template = Template(template.html_content)
                email_message.html_content = html_template.render(**variables)

                if template.text_content:
                    text_template = Template(template.text_content)
                    email_message.text_content = text_template.render(**variables)
            else:
                # 간단한 문자열 치환
                email_message.subject = template.subject.format(**variables)
                email_message.html_content = template.html_content.format(**variables)
                if template.text_content:
                    email_message.text_content = template.text_content.format(**variables)

        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise

    async def _send_via_smtp(self, email_message: EmailMessage):
        """SMTP를 통한 이메일 전송"""
        try:
            for recipient in email_message.recipients:
                # MIME 메시지 생성
                msg = MIMEMultipart('alternative')
                msg['Subject'] = email_message.subject
                msg['From'] = email_message.sender_email
                msg['To'] = recipient.email

                # 텍스트 및 HTML 내용 추가
                if email_message.text_content:
                    text_part = MIMEText(email_message.text_content, 'plain', 'utf-8')
                    msg.attach(text_part)

                if email_message.html_content:
                    html_part = MIMEText(email_message.html_content, 'html', 'utf-8')
                    msg.attach(html_part)

                # 첨부파일 추가
                for attachment in email_message.attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.content)
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment.filename}'
                    )
                    msg.attach(part)

                # SMTP 전송
                async with aiosmtplib.SMTP(
                    hostname=self.smtp_config["host"],
                    port=self.smtp_config["port"],
                    use_tls=self.smtp_config["use_tls"]
                ) as server:
                    if self.smtp_config["username"]:
                        await server.login(
                            self.smtp_config["username"],
                            self.smtp_config["password"]
                        )

                    await server.send_message(msg)

        except Exception as e:
            logger.error(f"SMTP send failed: {e}")
            raise

    async def _send_via_sendgrid(self, email_message: EmailMessage):
        """센드그리드를 통한 이메일 전송"""
        config = self.provider_config.get("sendgrid", {})
        api_key = config.get("api_key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        for recipient in email_message.recipients:
            payload = {
                "personalizations": [{
                    "to": [{"email": recipient.email, "name": recipient.name}],
                    "subject": email_message.subject
                }],
                "from": {"email": email_message.from_email, "name": email_message.from_name},
                "reply_to": {"email": email_message.reply_to or email_message.from_email},
                "content": []
            }

            if email_message.text_content:
                payload["content"].append({"type": "text/plain", "value": email_message.text_content})
            if email_message.html_content:
                payload["content"].append({"type": "text/html", "value": email_message.html_content})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status not in [200, 202]:
                        error = await response.text()
                        raise Exception(f"SendGrid error: {error}")
                    logger.info(f"Email sent via SendGrid to {recipient.email}")

    async def _send_via_aws_ses(self, email_message: EmailMessage):
        """아마존 SES를 통한 이메일 전송"""
        config = self.provider_config.get("aws_ses", {})
        region = config.get("region", "us-east-1")

        # boto3 사용
        try:
            import boto3
            from botocore.exceptions import ClientError

            ses_client = boto3.client(
                'ses',
                region_name=region,
                aws_access_key_id=config.get("access_key_id"),
                aws_secret_access_key=config.get("secret_access_key")
            )

            for recipient in email_message.recipients:
                try:
                    response = ses_client.send_email(
                        Source=f"{email_message.from_name} <{email_message.from_email}>",
                        Destination={'ToAddresses': [recipient.email]},
                        Message={
                            'Subject': {'Data': email_message.subject},
                            'Body': {
                                'Text': {'Data': email_message.text_content or ''},
                                'Html': {'Data': email_message.html_content or ''}
                            }
                        },
                        ReplyToAddresses=[email_message.reply_to] if email_message.reply_to else []
                    )
                    email_message.provider_message_id = response['MessageId']
                    logger.info(f"Email sent via AWS SES: {response['MessageId']}")
                except ClientError as e:
                    raise Exception(f"AWS SES error: {e.response['Error']['Message']}")
        except ImportError:
            # boto3가 설치되지 않은 경우 HTTP API 사용
            await self._send_via_aws_ses_http(email_message)

    async def _send_via_aws_ses_http(self, email_message: EmailMessage):
        """아마존 SES HTTP API를 통한 이메일 전송"""
        # AWS SES HTTP API 구현
        logger.warning("Using HTTP API for AWS SES (boto3 not available)")
        # 간단한 HTTP API 구현
        await self._send_via_generic_gateway(email_message)

    async def _send_via_mailgun(self, email_message: EmailMessage):
        """메일건을 통한 이메일 전송"""
        config = self.provider_config.get("mailgun", {})
        api_key = config.get("api_key")
        domain = config.get("domain")

        auth = aiohttp.BasicAuth("api", api_key)

        for recipient in email_message.recipients:
            data = {
                "from": f"{email_message.from_name} <{email_message.from_email}>",
                "to": recipient.email,
                "subject": email_message.subject,
                "text": email_message.text_content,
                "html": email_message.html_content
            }

            if email_message.reply_to:
                data["h:Reply-To"] = email_message.reply_to

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"https://api.mailgun.net/v3/{domain}/messages",
                    data=data,
                    auth=auth
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Mailgun error: {error}")
                    result = await response.json()
                    email_message.provider_message_id = result.get("id")
                    logger.info(f"Email sent via Mailgun: {result.get('id')}")

    async def _send_via_postmark(self, email_message: EmailMessage):
        """포스트마크를 통한 이메일 전송"""
        config = self.provider_config.get("postmark", {})
        api_key = config.get("api_key")

        headers = {
            "X-Postmark-Server-Token": api_key,
            "Content-Type": "application/json"
        }

        for recipient in email_message.recipients:
            payload = {
                "From": f"{email_message.from_name} <{email_message.from_email}>",
                "To": recipient.email,
                "Subject": email_message.subject,
                "TextBody": email_message.text_content,
                "HtmlBody": email_message.html_content,
                "ReplyTo": email_message.reply_to or email_message.from_email,
                "MessageStream": "outbound"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.postmarkapp.com/email",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Postmark error: {error}")
                    result = await response.json()
                    email_message.provider_message_id = result.get("MessageID")
                    logger.info(f"Email sent via Postmark: {result.get('MessageID')}")

    async def _send_via_mailjet(self, email_message: EmailMessage):
        """메일젯을 통한 이메일 전송"""
        config = self.provider_config.get("mailjet", {})

        auth = aiohttp.BasicAuth(
            config.get("api_key"),
            config.get("api_secret")
        )

        for recipient in email_message.recipients:
            payload = {
                "Messages": [{
                    "From": {
                        "Email": email_message.from_email,
                        "Name": email_message.from_name
                    },
                    "To": [{
                        "Email": recipient.email,
                        "Name": recipient.name
                    }],
                    "Subject": email_message.subject,
                    "TextPart": email_message.text_content,
                    "HTMLPart": email_message.html_content
                }]
            }

            if email_message.reply_to:
                payload["Messages"][0]["ReplyTo"] = {"Email": email_message.reply_to}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.mailjet.com/v3.1/send",
                    json=payload,
                    auth=auth
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Mailjet error: {error}")
                    result = await response.json()
                    if result["Messages"][0]["Status"] == "success":
                        email_message.provider_message_id = result["Messages"][0]["To"][0]["MessageID"]
                        logger.info(f"Email sent via Mailjet: {email_message.provider_message_id}")
                    else:
                        raise Exception(f"Mailjet error: {result}")

    async def _send_via_generic_gateway(self, email_message: EmailMessage):
        """일반 HTTP 이메일 게이트웨이를 통한 전송"""
        gateway_url = os.environ.get("EMAIL_GATEWAY_URL", "http://localhost:8080/email/send")
        gateway_token = os.environ.get("EMAIL_GATEWAY_TOKEN", "")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {gateway_token}"
        } if gateway_token else {"Content-Type": "application/json"}

        for recipient in email_message.recipients:
            payload = {
                "from": {
                    "email": email_message.from_email,
                    "name": email_message.from_name
                },
                "to": {
                    "email": recipient.email,
                    "name": recipient.name
                },
                "subject": email_message.subject,
                "text": email_message.text_content,
                "html": email_message.html_content,
                "reply_to": email_message.reply_to,
                "headers": email_message.headers,
                "metadata": email_message.metadata
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        gateway_url,
                        json=payload,
                        headers=headers,
                        timeout=30
                    ) as response:
                        if response.status in [200, 201, 202]:
                            result = await response.json()
                            email_message.provider_message_id = result.get("message_id", email_message.id)
                            logger.info(f"Email sent via generic gateway: {email_message.provider_message_id}")
                        else:
                            error = await response.text()
                            raise Exception(f"Generic gateway error ({response.status}): {error}")
            except asyncio.TimeoutError:
                raise Exception("Email gateway timeout after 30 seconds")

    async def _retry_processor(self):
        """재시도 처리 - 실제 구현"""
        max_retries = 3
        retry_delays = [60, 300, 900]  # 1분, 5분, 15분

        while True:
            try:
                await asyncio.sleep(30)  # 30초마다 체크

                # 실패한 이메일 찾기
                failed_emails = [
                    (msg_id, msg) for msg_id, msg in self.email_queue.items()
                    if msg.status == EmailStatus.FAILED and msg.retry_count < max_retries
                ]

                for msg_id, email_message in failed_emails:
                    # 재시도 대기 시간 확인
                    if email_message.last_retry_at:
                        time_since_retry = (datetime.now(timezone.utc) - email_message.last_retry_at).total_seconds()
                        required_delay = retry_delays[min(email_message.retry_count, len(retry_delays) - 1)]

                        if time_since_retry < required_delay:
                            continue  # 아직 대기 시간이 지나지 않음

                    # 재시도
                    try:
                        logger.info(f"Retrying email {msg_id} (attempt {email_message.retry_count + 1}/{max_retries})")
                        email_message.retry_count += 1
                        email_message.last_retry_at = datetime.now(timezone.utc)
                        email_message.status = EmailStatus.PENDING

                        await self._send_email_internal(email_message)

                        # 성공 시 큐에서 제거
                        del self.email_queue[msg_id]
                        self.delivery_stats["retried"] += 1

                    except Exception as e:
                        logger.error(f"Retry failed for {msg_id}: {e}")

                        if email_message.retry_count >= max_retries:
                            # 최대 재시도 횟수 초과
                            email_message.status = EmailStatus.BOUNCED
                            self.email_status[msg_id] = EmailStatus.BOUNCED
                            self.delivery_stats["bounced"] += 1
                            del self.email_queue[msg_id]
                            logger.error(f"Email {msg_id} marked as bounced after {max_retries} retries")

            except Exception as e:
                logger.error(f"Retry processor error: {e}")
                await asyncio.sleep(60)  # 오류 발생 시 1분 대기

    def get_email_status(self, email_id: str) -> Optional[EmailStatus]:
        """이메일 상태 조회"""
        return self.email_status.get(email_id)

    def get_delivery_stats(self) -> Dict[str, Any]:
        """전송 통계 조회"""
        total_emails = sum(self.delivery_stats.values())

        return {
            "total_emails": total_emails,
            "sent": self.delivery_stats["sent"],
            "delivered": self.delivery_stats["delivered"],
            "bounced": self.delivery_stats["bounced"],
            "failed": self.delivery_stats["failed"],
            "opened": self.delivery_stats["opened"],
            "clicked": self.delivery_stats["clicked"],
            "success_rate": (
                (self.delivery_stats["sent"] / total_emails * 100)
                if total_emails > 0 else 0
            ),
            "queue_size": len(self.email_queue)
        }

    async def shutdown(self):
        """서비스 종료"""
        try:
            # 백그라운드 작업 중지
            if self._processor_task:
                self._processor_task.cancel()
            if self._retry_task:
                self._retry_task.cancel()

            logger.info("Email service shutdown completed")

        except Exception as e:
            logger.error(f"Email service shutdown error: {e}")


# 전역 이메일 서비스 인스턴스
email_service: Optional[EmailService] = None

def get_email_service() -> Optional[EmailService]:
    """글로벌 이메일 서비스 반환"""
    return email_service

async def initialize_email_service(**kwargs) -> EmailService:
    """이메일 서비스 초기화"""
    global email_service

    email_service = EmailService(**kwargs)
    await email_service.initialize()

    return email_service