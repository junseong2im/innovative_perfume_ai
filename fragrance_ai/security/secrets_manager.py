"""
Secrets Management with KMS Integration
비밀/키 관리: .env 분리 + KMS/시크릿 매니저, 레포 노출 금지
"""

import os
import json
import base64
import hashlib
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class SecretProvider(str, Enum):
    """시크릿 제공자"""
    ENV = "env"  # Environment variables
    FILE = "file"  # .env file
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    AWS_KMS = "aws_kms"
    GCP_SECRET_MANAGER = "gcp_secret_manager"
    AZURE_KEY_VAULT = "azure_key_vault"
    HASHICORP_VAULT = "hashicorp_vault"


@dataclass
class SecretConfig:
    """시크릿 설정"""
    provider: SecretProvider
    region: Optional[str] = None
    vault_url: Optional[str] = None
    key_id: Optional[str] = None


class SecretsManager:
    """
    통합 시크릿 관리자

    지원 제공자:
    - Environment variables (.env)
    - AWS Secrets Manager
    - AWS KMS
    - GCP Secret Manager
    - Azure Key Vault
    - HashiCorp Vault
    """

    def __init__(self, config: Optional[SecretConfig] = None):
        self.config = config or SecretConfig(provider=SecretProvider.ENV)
        self._cache: Dict[str, str] = {}
        self._init_provider()

    def _init_provider(self):
        """제공자 초기화"""
        if self.config.provider == SecretProvider.AWS_SECRETS_MANAGER:
            self._init_aws_secrets_manager()
        elif self.config.provider == SecretProvider.AWS_KMS:
            self._init_aws_kms()
        elif self.config.provider == SecretProvider.GCP_SECRET_MANAGER:
            self._init_gcp_secret_manager()
        elif self.config.provider == SecretProvider.AZURE_KEY_VAULT:
            self._init_azure_key_vault()
        elif self.config.provider == SecretProvider.HASHICORP_VAULT:
            self._init_hashicorp_vault()

    def _init_aws_secrets_manager(self):
        """AWS Secrets Manager 초기화"""
        try:
            import boto3
            self.secrets_client = boto3.client(
                'secretsmanager',
                region_name=self.config.region or 'us-east-1'
            )
            logger.info("AWS Secrets Manager initialized")
        except ImportError:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            self.secrets_client = None
        except Exception as e:
            logger.error(f"Failed to initialize AWS Secrets Manager: {e}")
            self.secrets_client = None

    def _init_aws_kms(self):
        """AWS KMS 초기화"""
        try:
            import boto3
            self.kms_client = boto3.client(
                'kms',
                region_name=self.config.region or 'us-east-1'
            )
            logger.info("AWS KMS initialized")
        except ImportError:
            logger.warning("boto3 not installed. Install with: pip install boto3")
            self.kms_client = None
        except Exception as e:
            logger.error(f"Failed to initialize AWS KMS: {e}")
            self.kms_client = None

    def _init_gcp_secret_manager(self):
        """GCP Secret Manager 초기화"""
        try:
            from google.cloud import secretmanager
            self.gcp_client = secretmanager.SecretManagerServiceClient()
            logger.info("GCP Secret Manager initialized")
        except ImportError:
            logger.warning("google-cloud-secret-manager not installed")
            self.gcp_client = None
        except Exception as e:
            logger.error(f"Failed to initialize GCP Secret Manager: {e}")
            self.gcp_client = None

    def _init_azure_key_vault(self):
        """Azure Key Vault 초기화"""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient

            credential = DefaultAzureCredential()
            self.azure_client = SecretClient(
                vault_url=self.config.vault_url,
                credential=credential
            )
            logger.info("Azure Key Vault initialized")
        except ImportError:
            logger.warning("azure-keyvault-secrets not installed")
            self.azure_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Azure Key Vault: {e}")
            self.azure_client = None

    def _init_hashicorp_vault(self):
        """HashiCorp Vault 초기화"""
        try:
            import hvac
            self.vault_client = hvac.Client(url=self.config.vault_url)
            logger.info("HashiCorp Vault initialized")
        except ImportError:
            logger.warning("hvac not installed. Install with: pip install hvac")
            self.vault_client = None
        except Exception as e:
            logger.error(f"Failed to initialize HashiCorp Vault: {e}")
            self.vault_client = None

    def get_secret(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        시크릿 조회

        Args:
            secret_name: 시크릿 이름
            default: 기본값 (없는 경우)

        Returns:
            시크릿 값 또는 None
        """
        # Check cache first
        if secret_name in self._cache:
            return self._cache[secret_name]

        # Get from provider
        if self.config.provider == SecretProvider.ENV:
            value = self._get_from_env(secret_name)
        elif self.config.provider == SecretProvider.AWS_SECRETS_MANAGER:
            value = self._get_from_aws_secrets_manager(secret_name)
        elif self.config.provider == SecretProvider.AWS_KMS:
            value = self._get_from_aws_kms(secret_name)
        elif self.config.provider == SecretProvider.GCP_SECRET_MANAGER:
            value = self._get_from_gcp_secret_manager(secret_name)
        elif self.config.provider == SecretProvider.AZURE_KEY_VAULT:
            value = self._get_from_azure_key_vault(secret_name)
        elif self.config.provider == SecretProvider.HASHICORP_VAULT:
            value = self._get_from_hashicorp_vault(secret_name)
        else:
            value = None

        if value is None:
            value = default

        # Cache for reuse
        if value is not None:
            self._cache[secret_name] = value

        return value

    def _get_from_env(self, secret_name: str) -> Optional[str]:
        """환경 변수에서 조회"""
        return os.getenv(secret_name)

    def _get_from_aws_secrets_manager(self, secret_name: str) -> Optional[str]:
        """AWS Secrets Manager에서 조회"""
        if not self.secrets_client:
            logger.warning("AWS Secrets Manager not initialized, falling back to env")
            return self._get_from_env(secret_name)

        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            if 'SecretString' in response:
                return response['SecretString']
            else:
                # Binary secret
                return base64.b64decode(response['SecretBinary']).decode('utf-8')
        except self.secrets_client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret not found: {secret_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to get secret from AWS: {e}")
            return None

    def _get_from_aws_kms(self, secret_name: str) -> Optional[str]:
        """AWS KMS에서 복호화"""
        if not self.kms_client:
            logger.warning("AWS KMS not initialized, falling back to env")
            return self._get_from_env(secret_name)

        # Get encrypted value from env
        encrypted_value = os.getenv(f"{secret_name}_ENCRYPTED")
        if not encrypted_value:
            return None

        try:
            # Decode base64
            ciphertext_blob = base64.b64decode(encrypted_value)

            # Decrypt with KMS
            response = self.kms_client.decrypt(
                CiphertextBlob=ciphertext_blob,
                KeyId=self.config.key_id
            )

            return response['Plaintext'].decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decrypt with KMS: {e}")
            return None

    def _get_from_gcp_secret_manager(self, secret_name: str) -> Optional[str]:
        """GCP Secret Manager에서 조회"""
        if not self.gcp_client:
            logger.warning("GCP Secret Manager not initialized, falling back to env")
            return self._get_from_env(secret_name)

        try:
            project_id = os.getenv("GCP_PROJECT_ID")
            if not project_id:
                logger.error("GCP_PROJECT_ID not set")
                return None

            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = self.gcp_client.access_secret_version(request={"name": name})
            return response.payload.data.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get secret from GCP: {e}")
            return None

    def _get_from_azure_key_vault(self, secret_name: str) -> Optional[str]:
        """Azure Key Vault에서 조회"""
        if not self.azure_client:
            logger.warning("Azure Key Vault not initialized, falling back to env")
            return self._get_from_env(secret_name)

        try:
            secret = self.azure_client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to get secret from Azure: {e}")
            return None

    def _get_from_hashicorp_vault(self, secret_name: str) -> Optional[str]:
        """HashiCorp Vault에서 조회"""
        if not self.vault_client:
            logger.warning("HashiCorp Vault not initialized, falling back to env")
            return self._get_from_env(secret_name)

        try:
            # Assume KV v2 secrets engine at "secret/"
            path = f"secret/data/{secret_name}"
            response = self.vault_client.secrets.kv.v2.read_secret_version(path=secret_name)
            return response['data']['data'].get('value')
        except Exception as e:
            logger.error(f"Failed to get secret from Vault: {e}")
            return None

    def set_secret(self, secret_name: str, secret_value: str):
        """
        시크릿 설정 (지원되는 제공자만)

        주의: 프로덕션에서는 수동으로 설정 권장
        """
        if self.config.provider == SecretProvider.AWS_SECRETS_MANAGER:
            self._set_to_aws_secrets_manager(secret_name, secret_value)
        elif self.config.provider == SecretProvider.GCP_SECRET_MANAGER:
            self._set_to_gcp_secret_manager(secret_name, secret_value)
        elif self.config.provider == SecretProvider.AZURE_KEY_VAULT:
            self._set_to_azure_key_vault(secret_name, secret_value)
        elif self.config.provider == SecretProvider.HASHICORP_VAULT:
            self._set_to_hashicorp_vault(secret_name, secret_value)
        else:
            logger.warning(f"Set not supported for provider: {self.config.provider}")

    def _set_to_aws_secrets_manager(self, secret_name: str, secret_value: str):
        """AWS Secrets Manager에 저장"""
        if not self.secrets_client:
            return

        try:
            # Try to create or update
            try:
                self.secrets_client.create_secret(
                    Name=secret_name,
                    SecretString=secret_value
                )
                logger.info(f"Created secret: {secret_name}")
            except self.secrets_client.exceptions.ResourceExistsException:
                self.secrets_client.put_secret_value(
                    SecretId=secret_name,
                    SecretString=secret_value
                )
                logger.info(f"Updated secret: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to set secret in AWS: {e}")

    def _set_to_gcp_secret_manager(self, secret_name: str, secret_value: str):
        """GCP Secret Manager에 저장"""
        if not self.gcp_client:
            return

        try:
            project_id = os.getenv("GCP_PROJECT_ID")
            parent = f"projects/{project_id}"

            # Create secret if not exists
            try:
                self.gcp_client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_name,
                        "secret": {"replication": {"automatic": {}}}
                    }
                )
            except:
                pass  # Already exists

            # Add version
            secret_path = f"{parent}/secrets/{secret_name}"
            self.gcp_client.add_secret_version(
                request={
                    "parent": secret_path,
                    "payload": {"data": secret_value.encode('utf-8')}
                }
            )
            logger.info(f"Set secret in GCP: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to set secret in GCP: {e}")

    def _set_to_azure_key_vault(self, secret_name: str, secret_value: str):
        """Azure Key Vault에 저장"""
        if not self.azure_client:
            return

        try:
            self.azure_client.set_secret(secret_name, secret_value)
            logger.info(f"Set secret in Azure: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to set secret in Azure: {e}")

    def _set_to_hashicorp_vault(self, secret_name: str, secret_value: str):
        """HashiCorp Vault에 저장"""
        if not self.vault_client:
            return

        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=secret_name,
                secret={'value': secret_value}
            )
            logger.info(f"Set secret in Vault: {secret_name}")
        except Exception as e:
            logger.error(f"Failed to set secret in Vault: {e}")

    def validate_secrets(self, required_secrets: list) -> tuple[bool, list]:
        """
        필수 시크릿 존재 확인

        Returns:
            (all_present, missing_secrets)
        """
        missing = []

        for secret_name in required_secrets:
            value = self.get_secret(secret_name)
            if value is None:
                missing.append(secret_name)

        return len(missing) == 0, missing


# =============================================================================
# Artisan Required Secrets
# =============================================================================

REQUIRED_SECRETS = [
    # Database
    "DB_PASSWORD_PROD",
    "DB_PASSWORD_STG",

    # LLM API Keys
    "QWEN_API_KEY",
    "MISTRAL_API_KEY",
    "LLAMA_API_KEY",

    # AWS (if used)
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",

    # Monitoring
    "GRAFANA_API_KEY",

    # Encryption
    "JWT_SECRET_KEY",
    "ENCRYPTION_KEY",
]


# =============================================================================
# Global Instance
# =============================================================================

_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """글로벌 시크릿 관리자 반환"""
    global _secrets_manager

    if _secrets_manager is None:
        # Determine provider from environment
        provider_name = os.getenv("SECRETS_PROVIDER", "env").lower()
        provider = SecretProvider(provider_name)

        config = SecretConfig(
            provider=provider,
            region=os.getenv("AWS_REGION"),
            vault_url=os.getenv("VAULT_URL"),
            key_id=os.getenv("KMS_KEY_ID")
        )

        _secrets_manager = SecretsManager(config)

    return _secrets_manager


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """편의 함수: 시크릿 조회"""
    return get_secrets_manager().get_secret(name, default)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Secrets Manager")
    parser.add_argument("--validate", action="store_true", help="Validate required secrets")
    parser.add_argument("--get", help="Get secret by name")
    parser.add_argument("--set", nargs=2, metavar=("NAME", "VALUE"), help="Set secret")

    args = parser.parse_args()

    manager = get_secrets_manager()

    if args.validate:
        all_present, missing = manager.validate_secrets(REQUIRED_SECRETS)

        if all_present:
            print("✅ All required secrets are present")
        else:
            print(f"❌ Missing secrets: {', '.join(missing)}")
            exit(1)

    elif args.get:
        value = manager.get_secret(args.get)
        if value:
            print(f"{args.get}: [REDACTED - Length: {len(value)}]")
        else:
            print(f"{args.get}: NOT FOUND")

    elif args.set:
        name, value = args.set
        manager.set_secret(name, value)
        print(f"✅ Secret set: {name}")

    else:
        print("Use --validate, --get, or --set")
