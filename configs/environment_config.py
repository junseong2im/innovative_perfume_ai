"""
Environment Configuration
dev → stg → prod 환경 분리
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    host: str
    port: int
    database: str
    user: str
    password: str  # 실제로는 secrets에서 로드
    pool_size: int = 10


@dataclass
class LLMConfig:
    """LLM 설정"""
    qwen_endpoint: str
    mistral_endpoint: str
    llama_endpoint: str
    timeout: float = 30.0
    cache_ttl: int = 3600


@dataclass
class RLConfig:
    """RL 설정"""
    algorithm: str = "PPO"  # PPO or REINFORCE
    buffer_size: int = 2048
    learning_rate: float = 3e-4
    checkpoint_dir: str = "./checkpoints"


@dataclass
class EnvironmentConfig:
    """환경별 설정"""
    env: str
    debug: bool
    log_level: str
    database: DatabaseConfig
    llm: LLMConfig
    rl: RLConfig
    prometheus_url: str
    grafana_url: str


# =============================================================================
# Development Environment
# =============================================================================

DEV_CONFIG = EnvironmentConfig(
    env="dev",
    debug=True,
    log_level="DEBUG",
    database=DatabaseConfig(
        host="localhost",
        port=5432,
        database="artisan_dev",
        user="dev_user",
        password=os.getenv("DB_PASSWORD_DEV", "dev_password"),
        pool_size=5
    ),
    llm=LLMConfig(
        qwen_endpoint="http://localhost:8001/qwen",
        mistral_endpoint="http://localhost:8001/mistral",
        llama_endpoint="http://localhost:8001/llama",
        timeout=60.0,  # Longer timeout for dev
        cache_ttl=300  # 5 minutes for dev
    ),
    rl=RLConfig(
        algorithm="PPO",
        buffer_size=1024,  # Smaller for dev
        learning_rate=3e-4,
        checkpoint_dir="./checkpoints_dev"
    ),
    prometheus_url="http://localhost:9090",
    grafana_url="http://localhost:3000"
)


# =============================================================================
# Staging Environment
# =============================================================================

STG_CONFIG = EnvironmentConfig(
    env="stg",
    debug=False,
    log_level="INFO",
    database=DatabaseConfig(
        host="stg-db.artisan.internal",
        port=5432,
        database="artisan_stg",
        user="stg_user",
        password=os.getenv("DB_PASSWORD_STG", ""),
        pool_size=10
    ),
    llm=LLMConfig(
        qwen_endpoint="http://stg-llm.artisan.internal:8001/qwen",
        mistral_endpoint="http://stg-llm.artisan.internal:8001/mistral",
        llama_endpoint="http://stg-llm.artisan.internal:8001/llama",
        timeout=30.0,
        cache_ttl=1800  # 30 minutes for stg
    ),
    rl=RLConfig(
        algorithm="PPO",
        buffer_size=2048,
        learning_rate=3e-4,
        checkpoint_dir="/data/checkpoints_stg"
    ),
    prometheus_url="http://stg-prometheus.artisan.internal:9090",
    grafana_url="http://stg-grafana.artisan.internal:3000"
)


# =============================================================================
# Production Environment
# =============================================================================

PROD_CONFIG = EnvironmentConfig(
    env="prod",
    debug=False,
    log_level="WARNING",
    database=DatabaseConfig(
        host="prod-db.artisan.internal",
        port=5432,
        database="artisan_prod",
        user="prod_user",
        password=os.getenv("DB_PASSWORD_PROD", ""),
        pool_size=20
    ),
    llm=LLMConfig(
        qwen_endpoint="http://prod-llm.artisan.internal:8001/qwen",
        mistral_endpoint="http://prod-llm.artisan.internal:8001/mistral",
        llama_endpoint="http://prod-llm.artisan.internal:8001/llama",
        timeout=30.0,
        cache_ttl=3600  # 1 hour for prod
    ),
    rl=RLConfig(
        algorithm="PPO",
        buffer_size=2048,
        learning_rate=3e-4,
        checkpoint_dir="/data/checkpoints_prod"
    ),
    prometheus_url="http://prod-prometheus.artisan.internal:9090",
    grafana_url="http://prod-grafana.artisan.internal:3000"
)


# =============================================================================
# Configuration Loader
# =============================================================================

def get_config() -> EnvironmentConfig:
    """현재 환경 설정 반환"""
    env = os.getenv("ARTISAN_ENV", "dev").lower()

    config_map = {
        "dev": DEV_CONFIG,
        "stg": STG_CONFIG,
        "prod": PROD_CONFIG
    }

    config = config_map.get(env, DEV_CONFIG)

    # Validate secrets in non-dev environments
    if env != "dev":
        if not config.database.password:
            raise ValueError(f"DB_PASSWORD_{env.upper()} environment variable not set")

    return config


# =============================================================================
# Secrets Management
# =============================================================================

class SecretsManager:
    """
    시크릿 관리

    실제 환경에서는 HashiCorp Vault나 AWS Secrets Manager 사용
    로컬 개발 시에는 환경 변수 fallback
    """

    @staticmethod
    def get_secret(key: str, environment: str = "dev") -> str:
        """
        시크릿 조회

        우선순위:
        1. Vault API (VAULT_ADDR 설정된 경우)
        2. AWS Secrets Manager (AWS_SECRETS_ENABLED=true)
        3. 환경 변수 (fallback)
        """
        # Try Vault first
        vault_addr = os.getenv("VAULT_ADDR")
        if vault_addr:
            try:
                value = SecretsManager._get_from_vault(key, environment, vault_addr)
                if value:
                    return value
            except Exception as e:
                import logging
                logging.warning(f"Failed to get secret from Vault: {e}, falling back to env vars")

        # Try AWS Secrets Manager
        if os.getenv("AWS_SECRETS_ENABLED", "").lower() == "true":
            try:
                value = SecretsManager._get_from_aws_secrets(key, environment)
                if value:
                    return value
            except Exception as e:
                import logging
                logging.warning(f"Failed to get secret from AWS Secrets Manager: {e}, falling back to env vars")

        # Fallback to environment variables
        env_key = f"{key}_{environment.upper()}"
        value = os.getenv(env_key)

        if value is None:
            raise ValueError(f"Secret not found: {env_key} (tried Vault, AWS Secrets, and env vars)")

        return value

    @staticmethod
    def _get_from_vault(key: str, environment: str, vault_addr: str) -> str:
        """Vault에서 시크릿 조회"""
        import requests

        vault_token = os.getenv("VAULT_TOKEN")
        if not vault_token:
            raise ValueError("VAULT_TOKEN not set")

        # Vault KV v2 API
        secret_path = f"secret/data/{environment}/{key}"
        url = f"{vault_addr}/v1/{secret_path}"

        headers = {
            "X-Vault-Token": vault_token
        }

        response = requests.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()

        data = response.json()
        return data.get("data", {}).get("data", {}).get("value")

    @staticmethod
    def _get_from_aws_secrets(key: str, environment: str) -> str:
        """AWS Secrets Manager에서 시크릿 조회"""
        try:
            import boto3
            from botocore.exceptions import ClientError

            secret_name = f"{environment}/{key}"

            # AWS Secrets Manager 클라이언트
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )

            response = client.get_secret_value(SecretId=secret_name)

            # SecretString or SecretBinary
            if 'SecretString' in response:
                import json
                secret = json.loads(response['SecretString'])
                return secret.get('value')
            else:
                # Binary secret
                import base64
                return base64.b64decode(response['SecretBinary']).decode('utf-8')

        except ImportError:
            raise ValueError("boto3 not installed. Install with: pip install boto3")
        except ClientError as e:
            raise ValueError(f"Failed to get secret from AWS: {e}")

    @staticmethod
    def set_secret(key: str, value: str, environment: str = "dev"):
        """
        시크릿 설정 (개발/테스트용)

        프로덕션에서는 Vault CLI나 AWS Console 사용
        """
        # Vault가 설정되어 있으면 Vault에 저장
        vault_addr = os.getenv("VAULT_ADDR")
        if vault_addr:
            try:
                SecretsManager._set_to_vault(key, value, environment, vault_addr)
                return
            except Exception as e:
                import logging
                logging.warning(f"Failed to set secret in Vault: {e}, setting in env vars")

        # Fallback: 환경 변수에 설정 (테스트용)
        env_key = f"{key}_{environment.upper()}"
        os.environ[env_key] = value

    @staticmethod
    def _set_to_vault(key: str, value: str, environment: str, vault_addr: str):
        """Vault에 시크릿 저장"""
        import requests

        vault_token = os.getenv("VAULT_TOKEN")
        if not vault_token:
            raise ValueError("VAULT_TOKEN not set")

        # Vault KV v2 API
        secret_path = f"secret/data/{environment}/{key}"
        url = f"{vault_addr}/v1/{secret_path}"

        headers = {
            "X-Vault-Token": vault_token,
            "Content-Type": "application/json"
        }

        data = {
            "data": {
                "value": value
            }
        }

        response = requests.post(url, headers=headers, json=data, timeout=5.0)
        response.raise_for_status()

    @staticmethod
    def rotate_secret(key: str, environment: str = "prod"):
        """
        시크릿 로테이션 (프로덕션용)

        자동으로 새 비밀번호 생성하고 업데이트
        """
        import secrets
        import string

        # 안전한 랜덤 비밀번호 생성
        alphabet = string.ascii_letters + string.digits + string.punctuation
        new_password = ''.join(secrets.choice(alphabet) for _ in range(32))

        # 새 시크릿 저장
        SecretsManager.set_secret(key, new_password, environment)

        return new_password


# =============================================================================
# Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Environment Configuration")
    print("=" * 60)
    print()

    for env_name in ["dev", "stg", "prod"]:
        os.environ["ARTISAN_ENV"] = env_name
        config = get_config()

        print(f"Environment: {config.env}")
        print(f"  Debug: {config.debug}")
        print(f"  Log Level: {config.log_level}")
        print(f"  Database: {config.database.host}:{config.database.port}/{config.database.database}")
        print(f"  LLM Endpoint: {config.llm.qwen_endpoint}")
        print(f"  RL Algorithm: {config.rl.algorithm}")
        print(f"  Cache TTL: {config.llm.cache_ttl}s")
        print()
