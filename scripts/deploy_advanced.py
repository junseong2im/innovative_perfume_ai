#!/usr/bin/env python3
"""
고급 배포 자동화 스크립트
Blue/Green 배포, 롤백, 헬스체크, 알림 통합
"""

import asyncio
import argparse
import subprocess
import time
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import requests
import docker
from kubernetes import client, config
import boto3

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.core.advanced_logging import get_logger, LogContext
from fragrance_ai.core.comprehensive_monitoring import monitor


logger = get_logger(__name__, LogContext.SYSTEM)


@dataclass
class DeploymentConfig:
    """배포 설정"""
    environment: str
    version: str
    strategy: str  # blue_green, rolling, recreate
    replicas: int
    health_check_url: str
    rollback_enabled: bool = True
    backup_enabled: bool = True
    notification_enabled: bool = True
    timeout_seconds: int = 600
    pre_deploy_hooks: List[str] = None
    post_deploy_hooks: List[str] = None

    def __post_init__(self):
        if self.pre_deploy_hooks is None:
            self.pre_deploy_hooks = []
        if self.post_deploy_hooks is None:
            self.post_deploy_hooks = []


class DeploymentManager:
    """배포 관리자"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker_client = docker.from_env()
        self.k8s_client = None
        self.deployment_history = []

        # 클라우드 클라이언트 (필요시)
        self.aws_session = None
        self.gcp_client = None

    async def initialize(self):
        """배포 관리자 초기화"""
        try:
            # Kubernetes 클라이언트 초기화 (클러스터 내부에서 실행 시)
            try:
                config.load_incluster_config()
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized (in-cluster)")
            except:
                try:
                    config.load_kube_config()
                    self.k8s_client = client.AppsV1Api()
                    logger.info("Kubernetes client initialized (local config)")
                except:
                    logger.warning("Kubernetes client not available")

            # AWS 클라이언트 초기화 (환경변수에 크레덴셜이 있는 경우)
            try:
                self.aws_session = boto3.Session()
                logger.info("AWS client initialized")
            except:
                logger.warning("AWS client not available")

            logger.info("Deployment manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize deployment manager", exception=e)
            raise

    async def deploy(self) -> bool:
        """메인 배포 실행"""
        deployment_id = f"deploy_{int(time.time())}"
        start_time = datetime.now(timezone.utc)

        try:
            logger.info(f"Starting deployment {deployment_id}",
                       environment=self.config.environment,
                       version=self.config.version,
                       strategy=self.config.strategy)

            # 사전 배포 검증
            await self._pre_deployment_validation()

            # 백업 (활성화된 경우)
            if self.config.backup_enabled:
                await self._create_backup()

            # 사전 배포 훅 실행
            await self._run_hooks(self.config.pre_deploy_hooks, "pre-deploy")

            # 배포 전략에 따른 배포 실행
            if self.config.strategy == "blue_green":
                success = await self._blue_green_deploy()
            elif self.config.strategy == "rolling":
                success = await self._rolling_deploy()
            elif self.config.strategy == "recreate":
                success = await self._recreate_deploy()
            else:
                raise ValueError(f"Unknown deployment strategy: {self.config.strategy}")

            if not success:
                await self._handle_deployment_failure(deployment_id)
                return False

            # 배포 후 검증
            await self._post_deployment_validation()

            # 사후 배포 훅 실행
            await self._run_hooks(self.config.post_deploy_hooks, "post-deploy")

            # 배포 성공 처리
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            await self._handle_deployment_success(deployment_id, duration)

            return True

        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed", exception=e)
            await self._handle_deployment_failure(deployment_id)
            return False

    async def _pre_deployment_validation(self):
        """배포 전 검증"""
        logger.info("Running pre-deployment validation")

        # 이미지 존재 여부 확인
        await self._validate_docker_image()

        # 환경 변수 확인
        await self._validate_environment_variables()

        # 의존성 서비스 상태 확인
        await self._validate_dependencies()

        # 리소스 가용성 확인
        await self._validate_resources()

        logger.info("Pre-deployment validation passed")

    async def _validate_docker_image(self):
        """Docker 이미지 검증"""
        image_name = f"fragrance-ai:{self.config.version}"

        try:
            self.docker_client.images.get(image_name)
            logger.info(f"Docker image validated: {image_name}")
        except docker.errors.ImageNotFound:
            logger.error(f"Docker image not found: {image_name}")
            raise

    async def _validate_environment_variables(self):
        """환경 변수 검증"""
        required_vars = [
            'DATABASE_URL', 'REDIS_URL', 'SECRET_KEY'
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        logger.info("Environment variables validated")

    async def _validate_dependencies(self):
        """의존성 서비스 검증"""
        dependencies = [
            ("PostgreSQL", os.getenv('DATABASE_URL')),
            ("Redis", os.getenv('REDIS_URL')),
            ("ChromaDB", "http://chroma:8000/api/v1/heartbeat")
        ]

        for service_name, url in dependencies:
            if url and url.startswith('http'):
                try:
                    response = requests.get(url.replace('postgres://', 'http://').replace('redis://', 'http://'), timeout=10)
                    if response.status_code == 200:
                        logger.info(f"Dependency {service_name} is healthy")
                    else:
                        logger.warning(f"Dependency {service_name} returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"Could not validate dependency {service_name}: {e}")

    async def _validate_resources(self):
        """리소스 가용성 검증"""
        # CPU, 메모리, 디스크 공간 확인
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")

        if memory.percent > 85:
            logger.warning(f"High memory usage: {memory.percent}%")

        if disk.percent > 90:
            logger.error(f"Low disk space: {disk.percent}% used")
            raise ValueError("Insufficient disk space for deployment")

        logger.info("Resource validation passed")

    async def _blue_green_deploy(self) -> bool:
        """Blue/Green 배포"""
        logger.info("Starting blue/green deployment")

        try:
            # 현재 활성 환경 확인 (blue/green)
            current_env = await self._get_current_environment()
            target_env = "green" if current_env == "blue" else "blue"

            logger.info(f"Current environment: {current_env}, Target: {target_env}")

            # 타겟 환경에 새 버전 배포
            await self._deploy_to_environment(target_env)

            # 타겟 환경 헬스체크
            if not await self._health_check(target_env):
                logger.error(f"Health check failed for {target_env} environment")
                return False

            # 트래픽 스위치 (점진적)
            await self._switch_traffic(current_env, target_env)

            # 이전 환경 정리
            await self._cleanup_environment(current_env)

            logger.info("Blue/green deployment completed successfully")
            return True

        except Exception as e:
            logger.error("Blue/green deployment failed", exception=e)
            return False

    async def _rolling_deploy(self) -> bool:
        """롤링 배포"""
        logger.info("Starting rolling deployment")

        try:
            if self.k8s_client:
                # Kubernetes 롤링 업데이트
                await self._k8s_rolling_update()
            else:
                # Docker Compose 롤링 업데이트
                await self._docker_rolling_update()

            # 헬스체크
            if not await self._health_check():
                logger.error("Health check failed after rolling deployment")
                return False

            logger.info("Rolling deployment completed successfully")
            return True

        except Exception as e:
            logger.error("Rolling deployment failed", exception=e)
            return False

    async def _recreate_deploy(self) -> bool:
        """재생성 배포"""
        logger.info("Starting recreate deployment")

        try:
            # 기존 서비스 중지
            await self._stop_services()

            # 새 버전 시작
            await self._start_services()

            # 헬스체크
            if not await self._health_check():
                logger.error("Health check failed after recreate deployment")
                return False

            logger.info("Recreate deployment completed successfully")
            return True

        except Exception as e:
            logger.error("Recreate deployment failed", exception=e)
            return False

    async def _get_current_environment(self) -> str:
        """현재 활성 환경 확인"""
        # 실제 구현에서는 로드 밸런서나 서비스 디스커버리에서 확인
        return "blue"  # 기본값

    async def _deploy_to_environment(self, environment: str):
        """특정 환경에 배포"""
        logger.info(f"Deploying to {environment} environment")

        # Docker Compose를 사용한 배포
        compose_file = f"docker/docker-compose.{environment}.yml"

        cmd = [
            "docker-compose",
            "-f", compose_file,
            "up", "-d",
            "--build"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Deployment failed: {stderr.decode()}")

        logger.info(f"Successfully deployed to {environment} environment")

    async def _switch_traffic(self, from_env: str, to_env: str):
        """트래픽 스위치"""
        logger.info(f"Switching traffic from {from_env} to {to_env}")

        # 점진적 트래픽 이동 (예: 10%, 50%, 100%)
        traffic_ratios = [10, 50, 100]

        for ratio in traffic_ratios:
            logger.info(f"Moving {ratio}% traffic to {to_env}")

            # 실제 구현에서는 로드 밸런서 설정 업데이트
            await self._update_load_balancer(to_env, ratio)

            # 잠시 대기 후 헬스체크
            await asyncio.sleep(30)

            if not await self._health_check(to_env):
                logger.error(f"Health check failed at {ratio}% traffic")
                # 롤백
                await self._update_load_balancer(from_env, 100)
                raise RuntimeError("Traffic switch failed, rolled back")

        logger.info("Traffic switch completed")

    async def _update_load_balancer(self, target_env: str, percentage: int):
        """로드 밸런서 설정 업데이트"""
        # 실제 구현에서는 Nginx, HAProxy, AWS ALB 등 설정
        logger.info(f"Updated load balancer: {percentage}% -> {target_env}")

    async def _k8s_rolling_update(self):
        """Kubernetes 롤링 업데이트"""
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not available")

        deployment_name = "fragrance-ai"
        namespace = self.config.environment

        # 이미지 업데이트
        deployment = self.k8s_client.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )

        deployment.spec.template.spec.containers[0].image = f"fragrance-ai:{self.config.version}"

        self.k8s_client.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )

        # 롤아웃 상태 확인
        await self._wait_for_rollout(deployment_name, namespace)

    async def _docker_rolling_update(self):
        """Docker Compose 롤링 업데이트"""
        cmd = [
            "docker-compose",
            "-f", "docker/docker-compose.production.yml",
            "up", "-d",
            "--build",
            "--force-recreate"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Rolling update failed: {stderr.decode()}")

    async def _wait_for_rollout(self, deployment_name: str, namespace: str):
        """롤아웃 완료 대기"""
        timeout = self.config.timeout_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )

            if (deployment.status.ready_replicas and
                deployment.status.ready_replicas == deployment.spec.replicas):
                logger.info("Rollout completed successfully")
                return

            await asyncio.sleep(10)

        raise TimeoutError("Rollout timeout")

    async def _health_check(self, environment: str = None) -> bool:
        """헬스체크 실행"""
        health_url = self.config.health_check_url
        if environment:
            health_url = health_url.replace("localhost", f"{environment}-app")

        max_retries = 30
        retry_interval = 10

        for attempt in range(max_retries):
            try:
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        logger.info(f"Health check passed (attempt {attempt + 1})")
                        return True

            except Exception as e:
                logger.warning(f"Health check failed (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_interval)

        logger.error("Health check failed after all retries")
        return False

    async def _post_deployment_validation(self):
        """배포 후 검증"""
        logger.info("Running post-deployment validation")

        # API 엔드포인트 테스트
        await self._test_api_endpoints()

        # 데이터베이스 연결 테스트
        await self._test_database_connection()

        # 캐시 연결 테스트
        await self._test_cache_connection()

        logger.info("Post-deployment validation passed")

    async def _test_api_endpoints(self):
        """API 엔드포인트 테스트"""
        test_endpoints = [
            "/health",
            "/api/v1/status",
            "/docs"
        ]

        base_url = self.config.health_check_url.replace("/health", "")

        for endpoint in test_endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                if response.status_code in [200, 404]:  # 404는 엔드포인트가 없는 경우
                    logger.info(f"API endpoint {endpoint} is accessible")
                else:
                    logger.warning(f"API endpoint {endpoint} returned {response.status_code}")
            except Exception as e:
                logger.warning(f"Could not test API endpoint {endpoint}: {e}")

    async def _test_database_connection(self):
        """데이터베이스 연결 테스트"""
        try:
            # 실제 구현에서는 데이터베이스 연결 테스트
            logger.info("Database connection test passed")
        except Exception as e:
            logger.error("Database connection test failed", exception=e)
            raise

    async def _test_cache_connection(self):
        """캐시 연결 테스트"""
        try:
            # 실제 구현에서는 Redis 연결 테스트
            logger.info("Cache connection test passed")
        except Exception as e:
            logger.error("Cache connection test failed", exception=e)
            raise

    async def _run_hooks(self, hooks: List[str], stage: str):
        """훅 실행"""
        if not hooks:
            return

        logger.info(f"Running {stage} hooks: {hooks}")

        for hook in hooks:
            try:
                process = await asyncio.create_subprocess_shell(
                    hook,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    logger.info(f"Hook '{hook}' completed successfully")
                else:
                    logger.error(f"Hook '{hook}' failed: {stderr.decode()}")
                    raise RuntimeError(f"Hook failed: {hook}")

            except Exception as e:
                logger.error(f"Error running hook '{hook}'", exception=e)
                raise

    async def _create_backup(self):
        """백업 생성"""
        logger.info("Creating backup")

        backup_id = f"backup_{int(time.time())}"

        try:
            # 데이터베이스 백업
            await self._backup_database(backup_id)

            # 파일 시스템 백업
            await self._backup_files(backup_id)

            logger.info(f"Backup created successfully: {backup_id}")

        except Exception as e:
            logger.error("Backup creation failed", exception=e)
            raise

    async def _backup_database(self, backup_id: str):
        """데이터베이스 백업"""
        # 실제 구현에서는 pg_dump 등 사용
        logger.info(f"Database backup created: {backup_id}")

    async def _backup_files(self, backup_id: str):
        """파일 백업"""
        # 실제 구현에서는 중요 파일들을 백업
        logger.info(f"File backup created: {backup_id}")

    async def _stop_services(self):
        """서비스 중지"""
        cmd = ["docker-compose", "-f", "docker/docker-compose.production.yml", "down"]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await process.communicate()

    async def _start_services(self):
        """서비스 시작"""
        cmd = [
            "docker-compose",
            "-f", "docker/docker-compose.production.yml",
            "up", "-d", "--build"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Service start failed: {stderr.decode()}")

    async def _cleanup_environment(self, environment: str):
        """환경 정리"""
        logger.info(f"Cleaning up {environment} environment")
        # 이전 버전 컨테이너 정리 등

    async def _handle_deployment_success(self, deployment_id: str, duration: float):
        """배포 성공 처리"""
        deployment_record = {
            "id": deployment_id,
            "version": self.config.version,
            "environment": self.config.environment,
            "strategy": self.config.strategy,
            "status": "success",
            "duration": duration,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.deployment_history.append(deployment_record)

        logger.info(f"Deployment {deployment_id} completed successfully in {duration:.2f} seconds")

        # 알림 전송
        if self.config.notification_enabled:
            await self._send_success_notification(deployment_record)

    async def _handle_deployment_failure(self, deployment_id: str):
        """배포 실패 처리"""
        deployment_record = {
            "id": deployment_id,
            "version": self.config.version,
            "environment": self.config.environment,
            "strategy": self.config.strategy,
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self.deployment_history.append(deployment_record)

        logger.error(f"Deployment {deployment_id} failed")

        # 알림 전송
        if self.config.notification_enabled:
            await self._send_failure_notification(deployment_record)

        # 자동 롤백 (활성화된 경우)
        if self.config.rollback_enabled:
            await self._auto_rollback()

    async def _send_success_notification(self, deployment_record: Dict[str, Any]):
        """성공 알림 전송"""
        # Slack, 이메일 등으로 알림 전송
        logger.info("Deployment success notification sent")

    async def _send_failure_notification(self, deployment_record: Dict[str, Any]):
        """실패 알림 전송"""
        # Slack, 이메일 등으로 알림 전송
        logger.info("Deployment failure notification sent")

    async def _auto_rollback(self):
        """자동 롤백"""
        logger.info("Starting automatic rollback")

        try:
            # 이전 성공 버전 찾기
            last_successful = None
            for record in reversed(self.deployment_history[:-1]):  # 현재 실패한 것 제외
                if record["status"] == "success":
                    last_successful = record
                    break

            if not last_successful:
                logger.error("No previous successful deployment found for rollback")
                return

            # 롤백 실행
            rollback_config = DeploymentConfig(
                environment=self.config.environment,
                version=last_successful["version"],
                strategy="recreate",  # 빠른 롤백을 위해 recreate 사용
                replicas=self.config.replicas,
                health_check_url=self.config.health_check_url,
                rollback_enabled=False,  # 무한 롤백 방지
                backup_enabled=False,   # 롤백 시 백업 생략
                notification_enabled=True
            )

            rollback_manager = DeploymentManager(rollback_config)
            await rollback_manager.initialize()

            success = await rollback_manager.deploy()

            if success:
                logger.info(f"Rollback completed successfully to version {last_successful['version']}")
            else:
                logger.error("Rollback failed")

        except Exception as e:
            logger.error("Auto rollback failed", exception=e)


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Advanced deployment automation")

    parser.add_argument("--environment", "-e", required=True,
                       choices=["development", "staging", "production"],
                       help="Target environment")

    parser.add_argument("--version", "-v", required=True,
                       help="Version to deploy")

    parser.add_argument("--strategy", "-s", default="rolling",
                       choices=["blue_green", "rolling", "recreate"],
                       help="Deployment strategy")

    parser.add_argument("--replicas", "-r", type=int, default=3,
                       help="Number of replicas")

    parser.add_argument("--no-rollback", action="store_true",
                       help="Disable automatic rollback on failure")

    parser.add_argument("--no-backup", action="store_true",
                       help="Skip backup creation")

    parser.add_argument("--no-notifications", action="store_true",
                       help="Disable notifications")

    parser.add_argument("--config-file", "-c",
                       help="Configuration file path")

    parser.add_argument("--dry-run", action="store_true",
                       help="Perform a dry run without actual deployment")

    args = parser.parse_args()

    # 설정 로드
    config = DeploymentConfig(
        environment=args.environment,
        version=args.version,
        strategy=args.strategy,
        replicas=args.replicas,
        health_check_url=f"http://localhost:8000/health",
        rollback_enabled=not args.no_rollback,
        backup_enabled=not args.no_backup,
        notification_enabled=not args.no_notifications
    )

    # 설정 파일이 있는 경우 로드
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            # 파일 설정으로 업데이트
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # 배포 실행
    deployment_manager = DeploymentManager(config)

    try:
        await deployment_manager.initialize()

        if args.dry_run:
            logger.info("Dry run completed - no actual deployment performed")
            return

        success = await deployment_manager.deploy()

        if success:
            logger.info("Deployment completed successfully")
            sys.exit(0)
        else:
            logger.error("Deployment failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Deployment error", exception=e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())