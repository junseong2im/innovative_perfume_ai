#!/usr/bin/env python3
"""
Fragrance AI 데이터 백업 및 복구 시스템
자동화된 백업, 암호화, 다중 저장소 지원
"""

import os
import sys
import json
import gzip
import hashlib
import subprocess
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import boto3
import psycopg2
from cryptography.fernet import Fernet
import redis

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackupType(str, Enum):
    """백업 타입"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class StorageType(str, Enum):
    """스토리지 타입"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"

class BackupStatus(str, Enum):
    """백업 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"

@dataclass
class BackupConfig:
    """백업 설정"""
    name: str
    backup_type: BackupType
    storage_type: StorageType
    storage_config: Dict[str, Any]
    encryption_enabled: bool = True
    compression_enabled: bool = True
    retention_days: int = 30
    schedule_cron: Optional[str] = None

@dataclass
class BackupMetadata:
    """백업 메타데이터"""
    backup_id: str
    name: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    checksum: str
    storage_location: str
    encryption_key_id: Optional[str] = None
    status: BackupStatus = BackupStatus.COMPLETED
    error_message: Optional[str] = None
    source_info: Dict[str, Any] = None

class BackupEncryption:
    """백업 암호화 관리"""

    def __init__(self, key_file: str = "./keys/backup_encryption.key"):
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.encryption_key = self._get_or_create_key()

    def _get_or_create_key(self) -> bytes:
        """암호화 키 가져오기 또는 생성"""
        if self.key_file.exists():
            return self.key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            logger.info(f"새 암호화 키 생성: {self.key_file}")
            return key

    def encrypt_data(self, data: bytes) -> bytes:
        """데이터 암호화"""
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """데이터 복호화"""
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data)

    def encrypt_file(self, input_file: str, output_file: str):
        """파일 암호화"""
        with open(input_file, 'rb') as infile:
            data = infile.read()

        encrypted_data = self.encrypt_data(data)

        with open(output_file, 'wb') as outfile:
            outfile.write(encrypted_data)

    def decrypt_file(self, input_file: str, output_file: str):
        """파일 복호화"""
        with open(input_file, 'rb') as infile:
            encrypted_data = infile.read()

        decrypted_data = self.decrypt_data(encrypted_data)

        with open(output_file, 'wb') as outfile:
            outfile.write(decrypted_data)

class DatabaseBackupManager:
    """데이터베이스 백업 관리자"""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config

    async def backup_postgresql(
        self,
        output_file: str,
        backup_type: BackupType = BackupType.FULL,
        last_backup_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """PostgreSQL 백업"""
        try:
            # 백업 명령어 구성
            cmd = [
                "pg_dump",
                "-h", self.db_config["host"],
                "-p", str(self.db_config["port"]),
                "-U", self.db_config["username"],
                "-d", self.db_config["database"],
                "--no-password",
                "--verbose",
                "--format=custom",
                "--compress=9"
            ]

            # 증분 백업의 경우 (실제로는 WAL 기반 백업이 필요)
            if backup_type == BackupType.INCREMENTAL and last_backup_time:
                logger.warning("PostgreSQL incremental backup requires WAL-E or similar tools")

            # 환경 변수 설정
            env = os.environ.copy()
            env["PGPASSWORD"] = self.db_config["password"]

            # 백업 실행
            with open(output_file, 'wb') as f:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=f,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )

                _, stderr = await process.communicate()

                if process.returncode != 0:
                    raise Exception(f"pg_dump failed: {stderr.decode()}")

            # 파일 정보 수집
            file_size = os.path.getsize(output_file)
            checksum = self._calculate_checksum(output_file)

            return {
                "success": True,
                "file_size": file_size,
                "checksum": checksum,
                "backup_type": backup_type.value
            }

        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def backup_redis(self, output_file: str, redis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Redis 백업"""
        try:
            # Redis 연결
            r = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                password=redis_config.get("password"),
                db=redis_config.get("db", 0)
            )

            # RDB 백업 트리거
            r.bgsave()

            # 백업 완료 대기
            while r.lastsave() == r.lastsave():
                await asyncio.sleep(1)

            # RDB 파일 복사
            rdb_file = redis_config.get("rdb_path", "/var/lib/redis/dump.rdb")
            if os.path.exists(rdb_file):
                subprocess.run(["cp", rdb_file, output_file], check=True)
            else:
                raise Exception(f"Redis RDB file not found: {rdb_file}")

            file_size = os.path.getsize(output_file)
            checksum = self._calculate_checksum(output_file)

            return {
                "success": True,
                "file_size": file_size,
                "checksum": checksum
            }

        except Exception as e:
            logger.error(f"Redis backup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _calculate_checksum(self, file_path: str) -> str:
        """파일 체크섬 계산"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class FileBackupManager:
    """파일 시스템 백업 관리자"""

    def __init__(self):
        pass

    async def backup_directory(
        self,
        source_dir: str,
        output_file: str,
        backup_type: BackupType = BackupType.FULL,
        exclude_patterns: List[str] = None
    ) -> Dict[str, Any]:
        """디렉토리 백업"""
        try:
            exclude_patterns = exclude_patterns or [
                "*.pyc", "__pycache__", ".git", "node_modules", "*.log", "*.tmp"
            ]

            # tar 명령어 구성
            cmd = ["tar", "--create", "--gzip", "--file", output_file]

            # 제외 패턴 추가
            for pattern in exclude_patterns:
                cmd.extend(["--exclude", pattern])

            cmd.append(source_dir)

            # 백업 실행
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"tar backup failed: {stderr.decode()}")

            file_size = os.path.getsize(output_file)
            checksum = self._calculate_checksum(output_file)

            return {
                "success": True,
                "file_size": file_size,
                "checksum": checksum,
                "source_directory": source_dir
            }

        except Exception as e:
            logger.error(f"Directory backup failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _calculate_checksum(self, file_path: str) -> str:
        """파일 체크섬 계산"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

class StorageManager:
    """스토리지 관리자"""

    def __init__(self):
        self.storage_handlers = {
            StorageType.LOCAL: self._handle_local_storage,
            StorageType.S3: self._handle_s3_storage,
            StorageType.GCS: self._handle_gcs_storage,
            StorageType.AZURE: self._handle_azure_storage
        }

    async def upload_backup(
        self,
        local_file: str,
        storage_type: StorageType,
        storage_config: Dict[str, Any]
    ) -> str:
        """백업 파일 업로드"""
        handler = self.storage_handlers.get(storage_type)
        if not handler:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        return await handler(local_file, storage_config, action="upload")

    async def download_backup(
        self,
        remote_path: str,
        local_file: str,
        storage_type: StorageType,
        storage_config: Dict[str, Any]
    ):
        """백업 파일 다운로드"""
        handler = self.storage_handlers.get(storage_type)
        if not handler:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        await handler(local_file, storage_config, action="download", remote_path=remote_path)

    async def delete_backup(
        self,
        remote_path: str,
        storage_type: StorageType,
        storage_config: Dict[str, Any]
    ):
        """백업 파일 삭제"""
        handler = self.storage_handlers.get(storage_type)
        if not handler:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        await handler(None, storage_config, action="delete", remote_path=remote_path)

    async def _handle_local_storage(
        self,
        local_file: str,
        storage_config: Dict[str, Any],
        action: str,
        remote_path: str = None
    ) -> str:
        """로컬 스토리지 처리"""
        backup_dir = Path(storage_config["backup_directory"])
        backup_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(local_file).name
        destination = backup_dir / filename

        if action == "upload":
            subprocess.run(["cp", local_file, str(destination)], check=True)
            return str(destination)
        elif action == "download":
            subprocess.run(["cp", remote_path, local_file], check=True)
        elif action == "delete":
            if Path(remote_path).exists():
                Path(remote_path).unlink()

    async def _handle_s3_storage(
        self,
        local_file: str,
        storage_config: Dict[str, Any],
        action: str,
        remote_path: str = None
    ) -> str:
        """AWS S3 스토리지 처리"""
        s3_client = boto3.client(
            's3',
            aws_access_key_id=storage_config["access_key"],
            aws_secret_access_key=storage_config["secret_key"],
            region_name=storage_config.get("region", "us-east-1")
        )

        bucket_name = storage_config["bucket_name"]

        if action == "upload":
            key = f"backups/{Path(local_file).name}"
            s3_client.upload_file(local_file, bucket_name, key)
            return f"s3://{bucket_name}/{key}"
        elif action == "download":
            key = remote_path.replace(f"s3://{bucket_name}/", "")
            s3_client.download_file(bucket_name, key, local_file)
        elif action == "delete":
            key = remote_path.replace(f"s3://{bucket_name}/", "")
            s3_client.delete_object(Bucket=bucket_name, Key=key)

    async def _handle_gcs_storage(
        self,
        local_file: str,
        storage_config: Dict[str, Any],
        action: str,
        remote_path: str = None
    ) -> str:
        """Google Cloud Storage 처리 - 실제 구현"""
        try:
            from google.cloud import storage
            from google.oauth2 import service_account

            # 인증
            if 'credentials_path' in storage_config:
                credentials = service_account.Credentials.from_service_account_file(
                    storage_config['credentials_path']
                )
                client = storage.Client(credentials=credentials)
            else:
                client = storage.Client()  # 환경 변수 사용

            bucket_name = storage_config['bucket_name']
            bucket = client.bucket(bucket_name)

            if action == 'upload':
                # 업로드
                blob_name = os.path.basename(local_file)
                if 'path_prefix' in storage_config:
                    blob_name = f"{storage_config['path_prefix']}/{blob_name}"

                blob = bucket.blob(blob_name)

                # 멀티파트 업로드 (큰 파일)
                file_size = os.path.getsize(local_file)
                if file_size > 50 * 1024 * 1024:  # 50MB 이상
                    blob.chunk_size = 5 * 1024 * 1024  # 5MB chunks

                with open(local_file, 'rb') as f:
                    blob.upload_from_file(f)

                # 메타데이터 설정
                blob.metadata = {
                    'uploaded_at': datetime.now().isoformat(),
                    'original_name': os.path.basename(local_file),
                    'backup_type': 'fragrance_ai_backup'
                }
                blob.patch()

                logger.info(f"GCS upload completed: gs://{bucket_name}/{blob_name}")
                return f"gs://{bucket_name}/{blob_name}"

            elif action == 'download':
                # 다운로드
                blob = bucket.blob(remote_path)
                blob.download_to_filename(local_file)
                logger.info(f"GCS download completed: {remote_path} -> {local_file}")
                return local_file

            elif action == 'delete':
                # 삭제
                blob_name = remote_path.replace(f"gs://{bucket_name}/", "")
                blob = bucket.blob(blob_name)
                blob.delete()
                logger.info(f"GCS blob deleted: {remote_path}")

        except ImportError:
            logger.warning("google-cloud-storage not installed, using HTTP API fallback")
            return await self._handle_gcs_storage_http(local_file, storage_config, action, remote_path)
        except Exception as e:
            logger.error(f"GCS operation failed: {e}")
            raise

    async def _handle_gcs_storage_http(self, local_file: str, storage_config: Dict[str, Any], action: str, remote_path: str = None) -> str:
        """GCS HTTP API 폴백"""
        # OAuth2 토큰 발급
        import aiohttp
        import jwt
        import time

        # 서비스 계정 키로 JWT 생성
        if 'service_account_key' in storage_config:
            sa_key = storage_config['service_account_key']

            # JWT 클레임
            now = int(time.time())
            claim_set = {
                "iss": sa_key['client_email'],
                "scope": "https://www.googleapis.com/auth/devstorage.read_write",
                "aud": "https://oauth2.googleapis.com/token",
                "exp": now + 3600,
                "iat": now
            }

            # JWT 서명
            signed_jwt = jwt.encode(claim_set, sa_key['private_key'], algorithm='RS256')

            # 액세스 토큰 요청
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": signed_jwt
                    }
                ) as resp:
                    token_data = await resp.json()
                    access_token = token_data['access_token']

            # 실제 API 호출
            bucket_name = storage_config['bucket_name']

            if action == 'upload':
                blob_name = os.path.basename(local_file)
                upload_url = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={blob_name}"

                async with aiohttp.ClientSession() as session:
                    with open(local_file, 'rb') as f:
                        async with session.post(
                            upload_url,
                            data=f,
                            headers={'Authorization': f'Bearer {access_token}'}
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                return f"gs://{bucket_name}/{result['name']}"
                            else:
                                raise Exception(f"GCS HTTP upload failed: {await resp.text()}")

        return local_file

    async def _handle_azure_storage(
        self,
        local_file: str,
        storage_config: Dict[str, Any],
        action: str,
        remote_path: str = None
    ) -> str:
        """Azure Blob Storage 처리 - 실제 구현"""
        try:
            from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
            from azure.storage.blob import BlobBlock
            import uuid

            # 연결 문자열 또는 계정 키 사용
            if 'connection_string' in storage_config:
                blob_service_client = BlobServiceClient.from_connection_string(
                    storage_config['connection_string']
                )
            else:
                account_name = storage_config['account_name']
                account_key = storage_config['account_key']
                blob_service_client = BlobServiceClient(
                    account_url=f"https://{account_name}.blob.core.windows.net",
                    credential=account_key
                )

            container_name = storage_config['container_name']

            if action == 'upload':
                # 업로드
                blob_name = os.path.basename(local_file)
                if 'path_prefix' in storage_config:
                    blob_name = f"{storage_config['path_prefix']}/{blob_name}"

                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )

                file_size = os.path.getsize(local_file)

                # 큰 파일은 블록 업로드
                if file_size > 64 * 1024 * 1024:  # 64MB 이상
                    block_list = []
                    chunk_size = 4 * 1024 * 1024  # 4MB chunks

                    with open(local_file, 'rb') as f:
                        chunk_num = 0
                        while True:
                            chunk_data = f.read(chunk_size)
                            if not chunk_data:
                                break

                            block_id = str(uuid.uuid4())
                            blob_client.stage_block(block_id, chunk_data)
                            block_list.append(BlobBlock(block_id=block_id))
                            chunk_num += 1

                            if chunk_num % 10 == 0:
                                logger.info(f"Azure: Uploaded {chunk_num * chunk_size / 1024 / 1024:.1f}MB")

                    # 블록 커밋
                    blob_client.commit_block_list(block_list)
                else:
                    # 작은 파일은 한 번에 업로드
                    with open(local_file, 'rb') as f:
                        blob_client.upload_blob(f, overwrite=True)

                # 메타데이터 설정
                blob_client.set_blob_metadata({
                    'uploaded_at': datetime.now().isoformat(),
                    'original_name': os.path.basename(local_file),
                    'backup_type': 'fragrance_ai_backup'
                })

                logger.info(f"Azure upload completed: {container_name}/{blob_name}")
                return f"https://{storage_config.get('account_name', 'storage')}.blob.core.windows.net/{container_name}/{blob_name}"

            elif action == 'download':
                # 다운로드
                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=remote_path
                )

                with open(local_file, 'wb') as f:
                    download_stream = blob_client.download_blob()
                    f.write(download_stream.readall())

                logger.info(f"Azure download completed: {remote_path} -> {local_file}")
                return local_file

            elif action == 'delete':
                # 삭제
                blob_name = remote_path.split(f"{container_name}/", 1)[-1] if f"{container_name}/" in remote_path else remote_path
                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )
                blob_client.delete_blob()
                logger.info(f"Azure blob deleted: {remote_path}")

        except ImportError:
            logger.warning("azure-storage-blob not installed, using REST API fallback")
            return await self._handle_azure_storage_rest(local_file, storage_config, action, remote_path)
        except Exception as e:
            logger.error(f"Azure operation failed: {e}")
            raise

    async def _handle_azure_storage_rest(self, local_file: str, storage_config: Dict[str, Any], action: str, remote_path: str = None) -> str:
        """Azure REST API 폴백"""
        import aiohttp
        import hashlib
        import hmac
        import base64
        from datetime import datetime, timezone

        account_name = storage_config['account_name']
        account_key = storage_config['account_key']
        container_name = storage_config['container_name']

        if action == 'upload':
            blob_name = os.path.basename(local_file)

            # REST API 헤더 생성
            x_ms_date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
            x_ms_version = '2021-08-06'

            # 서명 문자열 생성
            string_to_sign = f"PUT\n\n\n{os.path.getsize(local_file)}\n\napplication/octet-stream\n\n\n\n\n\n\nx-ms-blob-type:BlockBlob\nx-ms-date:{x_ms_date}\nx-ms-version:{x_ms_version}\n/{account_name}/{container_name}/{blob_name}"

            # HMAC-SHA256 서명
            signature = base64.b64encode(
                hmac.new(
                    base64.b64decode(account_key),
                    string_to_sign.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')

            # HTTP 요청
            url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}"

            headers = {
                'x-ms-blob-type': 'BlockBlob',
                'x-ms-date': x_ms_date,
                'x-ms-version': x_ms_version,
                'Authorization': f"SharedKey {account_name}:{signature}",
                'Content-Type': 'application/octet-stream',
                'Content-Length': str(os.path.getsize(local_file))
            }

            async with aiohttp.ClientSession() as session:
                with open(local_file, 'rb') as f:
                    async with session.put(url, data=f, headers=headers) as resp:
                        if resp.status in [200, 201]:
                            logger.info(f"Azure REST upload completed: {url}")
                            return url
                        else:
                            error = await resp.text()
                            raise Exception(f"Azure REST upload failed: {error}")

        return local_file

class BackupManager:
    """메인 백업 관리자"""

    def __init__(self, config_file: str = "./configs/backup_config.json"):
        self.config_file = config_file
        self.backup_configs = self._load_configs()
        self.encryption = BackupEncryption()
        self.db_manager = None
        self.file_manager = FileBackupManager()
        self.storage_manager = StorageManager()
        self.metadata_store = {}

    def _load_configs(self) -> List[BackupConfig]:
        """백업 설정 로드"""
        if not os.path.exists(self.config_file):
            return self._create_default_configs()

        with open(self.config_file, 'r') as f:
            configs_data = json.load(f)

        configs = []
        for config_data in configs_data:
            config = BackupConfig(**config_data)
            configs.append(config)

        return configs

    def _create_default_configs(self) -> List[BackupConfig]:
        """기본 백업 설정 생성"""
        default_configs = [
            BackupConfig(
                name="postgresql_daily",
                backup_type=BackupType.FULL,
                storage_type=StorageType.LOCAL,
                storage_config={"backup_directory": "./backups/postgresql"},
                schedule_cron="0 2 * * *"  # 매일 오전 2시
            ),
            BackupConfig(
                name="redis_daily",
                backup_type=BackupType.FULL,
                storage_type=StorageType.LOCAL,
                storage_config={"backup_directory": "./backups/redis"},
                schedule_cron="0 3 * * *"  # 매일 오전 3시
            ),
            BackupConfig(
                name="application_data",
                backup_type=BackupType.FULL,
                storage_type=StorageType.LOCAL,
                storage_config={"backup_directory": "./backups/app_data"},
                schedule_cron="0 4 * * 0"  # 매주 일요일 오전 4시
            )
        ]

        # 설정 파일 저장
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump([asdict(config) for config in default_configs], f, indent=2)

        return default_configs

    async def create_backup(
        self,
        backup_name: str,
        sources: Dict[str, Any],
        storage_override: Optional[Dict[str, Any]] = None
    ) -> BackupMetadata:
        """백업 생성"""
        backup_id = f"{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"백업 시작: {backup_id}")

        try:
            # 임시 파일 경로
            temp_dir = Path("./temp_backups")
            temp_dir.mkdir(exist_ok=True)

            all_files = []
            total_size = 0

            # 각 소스별 백업 생성
            for source_type, source_config in sources.items():
                temp_file = temp_dir / f"{backup_id}_{source_type}.backup"

                if source_type == "postgresql":
                    if not self.db_manager:
                        self.db_manager = DatabaseBackupManager(source_config)

                    result = await self.db_manager.backup_postgresql(str(temp_file))

                elif source_type == "redis":
                    if not self.db_manager:
                        self.db_manager = DatabaseBackupManager({})

                    result = await self.db_manager.backup_redis(str(temp_file), source_config)

                elif source_type == "files":
                    result = await self.file_manager.backup_directory(
                        source_config["source_directory"],
                        str(temp_file),
                        exclude_patterns=source_config.get("exclude_patterns")
                    )
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    continue

                if result["success"]:
                    all_files.append({
                        "type": source_type,
                        "file": str(temp_file),
                        "size": result["file_size"],
                        "checksum": result["checksum"]
                    })
                    total_size += result["file_size"]
                else:
                    logger.error(f"Backup failed for {source_type}: {result['error']}")

            # 통합 아카이브 생성
            archive_file = temp_dir / f"{backup_id}.tar.gz"
            await self._create_archive(all_files, str(archive_file))

            # 암호화
            encrypted_file = None
            if self.encryption:
                encrypted_file = temp_dir / f"{backup_id}.encrypted"
                self.encryption.encrypt_file(str(archive_file), str(encrypted_file))
                final_file = str(encrypted_file)
            else:
                final_file = str(archive_file)

            # 스토리지에 업로드
            backup_config = next((config for config in self.backup_configs if config.name == backup_name), None)
            if backup_config:
                storage_config = storage_override or backup_config.storage_config
                storage_location = await self.storage_manager.upload_backup(
                    final_file,
                    backup_config.storage_type,
                    storage_config
                )
            else:
                storage_location = final_file

            # 메타데이터 생성
            metadata = BackupMetadata(
                backup_id=backup_id,
                name=backup_name,
                backup_type=BackupType.FULL,
                timestamp=datetime.now(),
                size_bytes=total_size,
                checksum=self._calculate_file_checksum(final_file),
                storage_location=storage_location,
                encryption_key_id="default" if encrypted_file else None,
                status=BackupStatus.COMPLETED,
                source_info=sources
            )

            # 메타데이터 저장
            await self._save_metadata(metadata)

            # 임시 파일 정리
            self._cleanup_temp_files(temp_dir)

            logger.info(f"백업 완료: {backup_id} ({total_size} bytes)")
            return metadata

        except Exception as e:
            logger.error(f"백업 실패: {e}")

            error_metadata = BackupMetadata(
                backup_id=backup_id,
                name=backup_name,
                backup_type=BackupType.FULL,
                timestamp=datetime.now(),
                size_bytes=0,
                checksum="",
                storage_location="",
                status=BackupStatus.FAILED,
                error_message=str(e)
            )

            await self._save_metadata(error_metadata)
            raise

    async def restore_backup(
        self,
        backup_id: str,
        restore_targets: Dict[str, Any],
        restore_point: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """백업 복원"""
        logger.info(f"백업 복원 시작: {backup_id}")

        try:
            # 메타데이터 로드
            metadata = await self._load_metadata(backup_id)
            if not metadata:
                raise Exception(f"Backup metadata not found: {backup_id}")

            # 백업 파일 다운로드
            temp_dir = Path("./temp_restore")
            temp_dir.mkdir(exist_ok=True)

            local_backup_file = temp_dir / f"{backup_id}.backup"

            # 스토리지에서 다운로드
            if metadata.storage_location:
                # 백업 구성 가져오기
                backup_config = next(
                    (config for config in self.backup_configs if config.name == metadata.name),
                    None
                )

                if backup_config:
                    # 스토리지에서 다운로드
                    await self.storage_manager.download_backup(
                        metadata.storage_location,
                        str(local_backup_file),
                        backup_config.storage_type,
                        backup_config.storage_config
                    )
                else:
                    # 로컬 파일로 가정
                    import shutil
                    if Path(metadata.storage_location).exists():
                        shutil.copy(metadata.storage_location, local_backup_file)
                    else:
                        raise FileNotFoundError(f"Backup file not found: {metadata.storage_location}")
            else:
                raise ValueError(f"No storage location in metadata for backup: {backup_id}")

            # 복호화
            if metadata.encryption_key_id:
                decrypted_file = temp_dir / f"{backup_id}_decrypted.backup"
                self.encryption.decrypt_file(str(local_backup_file), str(decrypted_file))
                local_backup_file = decrypted_file

            # 아카이브 추출
            extract_dir = temp_dir / f"{backup_id}_extracted"
            await self._extract_archive(str(local_backup_file), str(extract_dir))

            # 각 타겟별 복원
            restore_results = {}
            for target_type, target_config in restore_targets.items():
                result = await self._restore_target(
                    target_type, target_config, extract_dir, restore_point
                )
                restore_results[target_type] = result

            # 임시 파일 정리
            self._cleanup_temp_files(temp_dir)

            logger.info(f"백업 복원 완료: {backup_id}")
            return {
                "backup_id": backup_id,
                "restore_results": restore_results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"백업 복원 실패: {e}")
            raise

    async def _create_archive(self, files: List[Dict], output_file: str):
        """아카이브 생성"""
        cmd = ["tar", "--create", "--gzip", "--file", output_file]
        for file_info in files:
            cmd.append(file_info["file"])

        process = await asyncio.create_subprocess_exec(*cmd)
        await process.communicate()

        if process.returncode != 0:
            raise Exception("Failed to create archive")

    async def _extract_archive(self, archive_file: str, extract_dir: str):
        """아카이브 추출"""
        os.makedirs(extract_dir, exist_ok=True)

        cmd = ["tar", "--extract", "--gzip", "--file", archive_file, "--directory", extract_dir]
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.communicate()

        if process.returncode != 0:
            raise Exception("Failed to extract archive")

    async def _restore_target(
        self,
        target_type: str,
        target_config: Dict[str, Any],
        extract_dir: Path,
        restore_point: Optional[datetime]
    ) -> Dict[str, Any]:
        """특정 타겟 복원"""
        if target_type == "postgresql":
            return await self._restore_postgresql(target_config, extract_dir)
        elif target_type == "redis":
            return await self._restore_redis(target_config, extract_dir)
        elif target_type == "files":
            return await self._restore_files(target_config, extract_dir)
        else:
            return {"success": False, "error": f"Unknown target type: {target_type}"}

    async def _restore_postgresql(self, config: Dict[str, Any], extract_dir: Path) -> Dict[str, Any]:
        """PostgreSQL 복원"""
        try:
            backup_files = list(extract_dir.glob("*postgresql*.backup"))
            if not backup_files:
                return {"success": False, "error": "PostgreSQL backup file not found"}

            backup_file = backup_files[0]

            cmd = [
                "pg_restore",
                "-h", config["host"],
                "-p", str(config["port"]),
                "-U", config["username"],
                "-d", config["database"],
                "--clean",
                "--no-owner",
                "--verbose",
                str(backup_file)
            ]

            env = os.environ.copy()
            env["PGPASSWORD"] = config["password"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await process.communicate()

            if process.returncode != 0:
                return {"success": False, "error": stderr.decode()}

            return {"success": True, "restored_file": str(backup_file)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _restore_redis(self, config: Dict[str, Any], extract_dir: Path) -> Dict[str, Any]:
        """Redis 복원"""
        try:
            backup_files = list(extract_dir.glob("*redis*.backup"))
            if not backup_files:
                return {"success": False, "error": "Redis backup file not found"}

            backup_file = backup_files[0]
            rdb_path = config.get("rdb_path", "/var/lib/redis/dump.rdb")

            # Redis 서비스 중지 필요
            logger.warning("Redis restore requires service restart")

            subprocess.run(["cp", str(backup_file), rdb_path], check=True)

            return {"success": True, "restored_file": str(backup_file)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _restore_files(self, config: Dict[str, Any], extract_dir: Path) -> Dict[str, Any]:
        """파일 시스템 복원"""
        try:
            backup_files = list(extract_dir.glob("*files*.backup"))
            if not backup_files:
                return {"success": False, "error": "Files backup not found"}

            backup_file = backup_files[0]
            target_dir = config["target_directory"]

            # 백업에서 파일 추출
            cmd = ["tar", "--extract", "--gzip", "--file", str(backup_file), "--directory", target_dir]
            process = await asyncio.create_subprocess_exec(*cmd)
            await process.communicate()

            if process.returncode != 0:
                return {"success": False, "error": "Failed to extract files"}

            return {"success": True, "restored_to": target_dir}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _save_metadata(self, metadata: BackupMetadata):
        """메타데이터 저장"""
        metadata_dir = Path("./backups/metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = metadata_dir / f"{metadata.backup_id}.json"

        with open(metadata_file, 'w') as f:
            # datetime 객체를 문자열로 변환
            metadata_dict = asdict(metadata)
            metadata_dict["timestamp"] = metadata.timestamp.isoformat()
            json.dump(metadata_dict, f, indent=2)

    async def _load_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """메타데이터 로드"""
        metadata_file = Path(f"./backups/metadata/{backup_id}.json")

        if not metadata_file.exists():
            return None

        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)

        # 문자열을 datetime 객체로 변환
        metadata_dict["timestamp"] = datetime.fromisoformat(metadata_dict["timestamp"])

        return BackupMetadata(**metadata_dict)

    def _calculate_file_checksum(self, file_path: str) -> str:
        """파일 체크섬 계산"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _cleanup_temp_files(self, temp_dir: Path):
        """임시 파일 정리"""
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    async def list_backups(self) -> List[BackupMetadata]:
        """백업 목록 조회"""
        metadata_dir = Path("./backups/metadata")
        if not metadata_dir.exists():
            return []

        backups = []
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                backup_id = metadata_file.stem
                metadata = await self._load_metadata(backup_id)
                if metadata:
                    backups.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load metadata for {metadata_file}: {e}")

        return sorted(backups, key=lambda b: b.timestamp, reverse=True)

    async def cleanup_old_backups(self, retention_days: int = 30):
        """오래된 백업 정리"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        backups = await self.list_backups()

        for backup in backups:
            if backup.timestamp < cutoff_date:
                logger.info(f"Cleaning up old backup: {backup.backup_id}")

                # 메타데이터 파일 삭제
                metadata_file = Path(f"./backups/metadata/{backup.backup_id}.json")
                if metadata_file.exists():
                    metadata_file.unlink()

                # 스토리지에서 백업 파일 삭제
                if backup.storage_location:
                    # 백업 구성 가져오기
                    backup_config = next(
                        (config for config in self.backup_configs if config.name == backup.name),
                        None
                    )

                    if backup_config:
                        # 스토리지에서 삭제
                        await self.storage_manager.delete_backup(
                            backup.storage_location,
                            backup_config.storage_type,
                            backup_config.storage_config
                        )
                        logger.info(f"Deleted backup from storage: {backup.storage_location}")
                    else:
                        # 로컬 파일 삭제
                        local_path = Path(backup.storage_location)
                        if local_path.exists():
                            local_path.unlink()
                            logger.info(f"Deleted local backup: {backup.storage_location}")

async def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Fragrance AI Backup System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 백업 명령어
    backup_parser = subparsers.add_parser('backup', help='Create backup')
    backup_parser.add_argument('--name', required=True, help='Backup name')
    backup_parser.add_argument('--config', help='Custom config file')

    # 복원 명령어
    restore_parser = subparsers.add_parser('restore', help='Restore backup')
    restore_parser.add_argument('--backup-id', required=True, help='Backup ID to restore')
    restore_parser.add_argument('--config', help='Restore config file')

    # 목록 명령어
    list_parser = subparsers.add_parser('list', help='List backups')

    # 정리 명령어
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old backups')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Retention days')

    args = parser.parse_args()

    backup_manager = BackupManager()

    if args.command == 'backup':
        sources = {
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "username": "fragrance_ai",
                "password": "fragrance_ai_password",
                "database": "fragrance_ai"
            },
            "files": {
                "source_directory": "./fragrance_ai",
                "exclude_patterns": ["*.pyc", "__pycache__", "*.log"]
            }
        }

        metadata = await backup_manager.create_backup(args.name, sources)
        print(f"백업 완료: {metadata.backup_id}")

    elif args.command == 'restore':
        restore_targets = {
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "username": "fragrance_ai",
                "password": "fragrance_ai_password",
                "database": "fragrance_ai_restore"
            }
        }

        result = await backup_manager.restore_backup(args.backup_id, restore_targets)
        print(f"복원 완료: {result['backup_id']}")

    elif args.command == 'list':
        backups = await backup_manager.list_backups()
        print(f"총 {len(backups)}개의 백업:")
        for backup in backups[:10]:  # 최신 10개만 표시
            print(f"  {backup.backup_id} - {backup.timestamp} ({backup.size_bytes} bytes)")

    elif args.command == 'cleanup':
        await backup_manager.cleanup_old_backups(args.days)
        print(f"{args.days}일 이상된 백업 정리 완료")

    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())