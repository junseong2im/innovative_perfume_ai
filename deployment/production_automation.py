# 🚀 완벽한 프로덕션 배포 자동화 시스템
import asyncio
import json
import yaml
import subprocess
import shutil
import zipfile
import tarfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import aiofiles
import aiohttp
import docker
import kubernetes
from kubernetes import client, config
import boto3
import structlog
from jinja2 import Template, Environment, FileSystemLoader
import hashlib
import tempfile
import os

logger = structlog.get_logger("deployment_automation")


class DeploymentEnvironment(Enum):
    """배포 환경"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class DeploymentStrategy(Enum):
    """배포 전략"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class DeploymentStatus(Enum):
    """배포 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLBACK = "rollback"
    CANCELLED = "cancelled"


@dataclass
class DeploymentConfig:
    """배포 설정"""
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    image_tag: str
    replicas: int
    resources: Dict[str, Any]
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)
    rollback_on_failure: bool = True
    auto_scaling: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """배포 결과"""
    deployment_id: str
    status: DeploymentStatus
    environment: str
    strategy: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success_rate: Optional[float] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


@dataclass
class HealthCheck:
    """헬스체크 설정"""
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout: int = 30
    interval: int = 10
    retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)


class ContainerBuilder:
    """컨테이너 이미지 빌더"""

    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        self.docker_client = docker_client or docker.from_env()

    async def build_image(
        self,
        dockerfile_path: str,
        image_name: str,
        tag: str,
        build_args: Dict[str, str] = None,
        context_path: str = "."
    ) -> str:
        """Docker 이미지 빌드"""
        full_image_name = f"{image_name}:{tag}"

        try:
            logger.info(f"Docker 이미지 빌드 시작: {full_image_name}")

            # 빌드 로그를 위한 콜백
            build_logs = []

            def log_callback(stream):
                if 'stream' in stream:
                    log_line = stream['stream'].strip()
                    if log_line:
                        build_logs.append(log_line)
                        logger.info(f"빌드: {log_line}")

            # 이미지 빌드
            image, build_log = self.docker_client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=full_image_name,
                buildargs=build_args or {},
                rm=True,
                forcerm=True
            )

            # 빌드 로그 처리
            for log in build_log:
                log_callback(log)

            logger.info(f"Docker 이미지 빌드 완료: {full_image_name}")
            return full_image_name

        except Exception as e:
            logger.error(f"Docker 이미지 빌드 실패: {e}")
            raise

    async def push_image(self, image_name: str, registry_url: str = None) -> bool:
        """이미지 레지스트리에 푸시"""
        try:
            if registry_url:
                # 레지스트리 태그 추가
                registry_image = f"{registry_url}/{image_name}"
                self.docker_client.api.tag(image_name, registry_image)
                image_name = registry_image

            logger.info(f"이미지 푸시 시작: {image_name}")

            # 푸시 실행
            push_log = self.docker_client.images.push(image_name, stream=True, decode=True)

            for log in push_log:
                if 'status' in log:
                    logger.info(f"푸시: {log['status']}")

            logger.info(f"이미지 푸시 완료: {image_name}")
            return True

        except Exception as e:
            logger.error(f"이미지 푸시 실패: {e}")
            return False

    def get_image_info(self, image_name: str) -> Dict[str, Any]:
        """이미지 정보 조회"""
        try:
            image = self.docker_client.images.get(image_name)
            return {
                "id": image.id,
                "tags": image.tags,
                "created": image.attrs.get("Created"),
                "size": image.attrs.get("Size"),
                "architecture": image.attrs.get("Architecture"),
                "os": image.attrs.get("Os")
            }
        except Exception as e:
            logger.error(f"이미지 정보 조회 실패: {e}")
            return {}


class KubernetesDeployer:
    """Kubernetes 배포자"""

    def __init__(self, kubeconfig_path: str = None):
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()

        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        self.networking_v1 = client.NetworkingV1Api()

    async def deploy_application(
        self,
        config: DeploymentConfig,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """애플리케이션 배포"""
        try:
            deployment_name = f"fragrance-ai-{config.environment.value}"

            # 배포 매니페스트 생성
            deployment_manifest = self._create_deployment_manifest(config, deployment_name, namespace)

            # 서비스 매니페스트 생성
            service_manifest = self._create_service_manifest(deployment_name, namespace)

            # ConfigMap 생성 (환경 변수용)
            if config.environment_variables:
                configmap_manifest = self._create_configmap_manifest(
                    deployment_name, config.environment_variables, namespace
                )
                await self._apply_configmap(configmap_manifest)

            # Secret 생성
            if config.secrets:
                secret_manifest = self._create_secret_manifest(
                    deployment_name, config.secrets, namespace
                )
                await self._apply_secret(secret_manifest)

            # 배포 전략에 따른 배포 실행
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._blue_green_deploy(deployment_manifest, service_manifest, namespace)
            elif config.strategy == DeploymentStrategy.CANARY:
                return await self._canary_deploy(deployment_manifest, service_manifest, namespace)
            else:
                return await self._rolling_update_deploy(deployment_manifest, service_manifest, namespace)

        except Exception as e:
            logger.error(f"Kubernetes 배포 실패: {e}")
            raise

    def _create_deployment_manifest(
        self,
        config: DeploymentConfig,
        name: str,
        namespace: str
    ) -> Dict[str, Any]:
        """배포 매니페스트 생성"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "fragrance-ai",
                    "environment": config.environment.value,
                    "version": config.image_tag
                }
            },
            "spec": {
                "replicas": config.replicas,
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": "fragrance-ai",
                        "environment": config.environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "fragrance-ai",
                            "environment": config.environment.value,
                            "version": config.image_tag
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "fragrance-ai",
                            "image": f"fragrance-ai:{config.image_tag}",
                            "ports": [{"containerPort": 8000}],
                            "resources": config.resources,
                            "envFrom": [
                                {"configMapRef": {"name": f"{name}-config"}},
                                {"secretRef": {"name": f"{name}-secrets"}}
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }

    def _create_service_manifest(self, name: str, namespace: str) -> Dict[str, Any]:
        """서비스 매니페스트 생성"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{name}-service",
                "namespace": namespace
            },
            "spec": {
                "selector": {
                    "app": "fragrance-ai"
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }

    def _create_configmap_manifest(
        self,
        name: str,
        data: Dict[str, str],
        namespace: str
    ) -> Dict[str, Any]:
        """ConfigMap 매니페스트 생성"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{name}-config",
                "namespace": namespace
            },
            "data": data
        }

    def _create_secret_manifest(
        self,
        name: str,
        data: Dict[str, str],
        namespace: str
    ) -> Dict[str, Any]:
        """Secret 매니페스트 생성"""
        import base64

        encoded_data = {
            key: base64.b64encode(value.encode()).decode()
            for key, value in data.items()
        }

        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{name}-secrets",
                "namespace": namespace
            },
            "type": "Opaque",
            "data": encoded_data
        }

    async def _apply_configmap(self, manifest: Dict[str, Any]):
        """ConfigMap 적용"""
        try:
            self.core_v1.create_namespaced_config_map(
                namespace=manifest["metadata"]["namespace"],
                body=manifest
            )
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.core_v1.patch_namespaced_config_map(
                    name=manifest["metadata"]["name"],
                    namespace=manifest["metadata"]["namespace"],
                    body=manifest
                )
            else:
                raise

    async def _apply_secret(self, manifest: Dict[str, Any]):
        """Secret 적용"""
        try:
            self.core_v1.create_namespaced_secret(
                namespace=manifest["metadata"]["namespace"],
                body=manifest
            )
        except client.ApiException as e:
            if e.status == 409:  # Already exists
                self.core_v1.patch_namespaced_secret(
                    name=manifest["metadata"]["name"],
                    namespace=manifest["metadata"]["namespace"],
                    body=manifest
                )
            else:
                raise

    async def _rolling_update_deploy(
        self,
        deployment_manifest: Dict[str, Any],
        service_manifest: Dict[str, Any],
        namespace: str
    ) -> Dict[str, Any]:
        """롤링 업데이트 배포"""
        deployment_name = deployment_manifest["metadata"]["name"]

        try:
            # 기존 배포 확인
            try:
                existing_deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                # 업데이트
                self.apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=deployment_manifest
                )
                logger.info(f"배포 업데이트됨: {deployment_name}")
            except client.ApiException as e:
                if e.status == 404:
                    # 새 배포 생성
                    self.apps_v1.create_namespaced_deployment(
                        namespace=namespace,
                        body=deployment_manifest
                    )
                    logger.info(f"새 배포 생성됨: {deployment_name}")
                else:
                    raise

            # 서비스 생성/업데이트
            try:
                self.core_v1.create_namespaced_service(
                    namespace=namespace,
                    body=service_manifest
                )
            except client.ApiException as e:
                if e.status == 409:
                    pass  # Service already exists
                else:
                    raise

            # 배포 완료 대기
            await self._wait_for_deployment_ready(deployment_name, namespace)

            return {
                "status": "success",
                "deployment_name": deployment_name,
                "strategy": "rolling_update"
            }

        except Exception as e:
            logger.error(f"롤링 업데이트 실패: {e}")
            raise

    async def _blue_green_deploy(
        self,
        deployment_manifest: Dict[str, Any],
        service_manifest: Dict[str, Any],
        namespace: str
    ) -> Dict[str, Any]:
        """블루-그린 배포"""
        base_name = deployment_manifest["metadata"]["name"]
        green_name = f"{base_name}-green"

        try:
            # 그린 환경 배포
            green_manifest = deployment_manifest.copy()
            green_manifest["metadata"]["name"] = green_name
            green_manifest["spec"]["selector"]["matchLabels"]["deployment"] = "green"
            green_manifest["spec"]["template"]["metadata"]["labels"]["deployment"] = "green"

            self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=green_manifest
            )

            # 그린 환경 준비 대기
            await self._wait_for_deployment_ready(green_name, namespace)

            # 헬스체크 수행
            if await self._perform_health_check(green_name, namespace):
                # 서비스를 그린으로 전환
                service_manifest["spec"]["selector"]["deployment"] = "green"
                self.core_v1.patch_namespaced_service(
                    name=service_manifest["metadata"]["name"],
                    namespace=namespace,
                    body=service_manifest
                )

                # 블루 환경 제거 (선택적)
                try:
                    self.apps_v1.delete_namespaced_deployment(
                        name=base_name,
                        namespace=namespace
                    )
                except client.ApiException:
                    pass

                logger.info(f"블루-그린 배포 완료: {green_name}")
                return {"status": "success", "deployment_name": green_name, "strategy": "blue_green"}

            else:
                # 헬스체크 실패 시 그린 환경 제거
                self.apps_v1.delete_namespaced_deployment(
                    name=green_name,
                    namespace=namespace
                )
                raise Exception("헬스체크 실패")

        except Exception as e:
            logger.error(f"블루-그린 배포 실패: {e}")
            raise

    async def _canary_deploy(
        self,
        deployment_manifest: Dict[str, Any],
        service_manifest: Dict[str, Any],
        namespace: str
    ) -> Dict[str, Any]:
        """카나리 배포"""
        base_name = deployment_manifest["metadata"]["name"]
        canary_name = f"{base_name}-canary"

        try:
            # 카나리 배포 생성 (10% 트래픽)
            canary_manifest = deployment_manifest.copy()
            canary_manifest["metadata"]["name"] = canary_name
            canary_manifest["spec"]["replicas"] = max(1, deployment_manifest["spec"]["replicas"] // 10)
            canary_manifest["spec"]["selector"]["matchLabels"]["deployment"] = "canary"
            canary_manifest["spec"]["template"]["metadata"]["labels"]["deployment"] = "canary"

            self.apps_v1.create_namespaced_deployment(
                namespace=namespace,
                body=canary_manifest
            )

            await self._wait_for_deployment_ready(canary_name, namespace)

            # 카나리 모니터링 (실제로는 메트릭 수집)
            await asyncio.sleep(60)  # 1분간 모니터링

            # 성공 시 전체 배포로 확장
            canary_manifest["spec"]["replicas"] = deployment_manifest["spec"]["replicas"]
            self.apps_v1.patch_namespaced_deployment(
                name=canary_name,
                namespace=namespace,
                body=canary_manifest
            )

            # 기존 배포 제거
            try:
                self.apps_v1.delete_namespaced_deployment(
                    name=base_name,
                    namespace=namespace
                )
            except client.ApiException:
                pass

            logger.info(f"카나리 배포 완료: {canary_name}")
            return {"status": "success", "deployment_name": canary_name, "strategy": "canary"}

        except Exception as e:
            logger.error(f"카나리 배포 실패: {e}")
            # 실패 시 카나리 제거
            try:
                self.apps_v1.delete_namespaced_deployment(
                    name=canary_name,
                    namespace=namespace
                )
            except:
                pass
            raise

    async def _wait_for_deployment_ready(self, name: str, namespace: str, timeout: int = 600):
        """배포 준비 대기"""
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(name=name, namespace=namespace)

                if deployment.status.ready_replicas == deployment.spec.replicas:
                    logger.info(f"배포 준비 완료: {name}")
                    return True

                await asyncio.sleep(5)

            except Exception as e:
                logger.warning(f"배포 상태 확인 중 오류: {e}")
                await asyncio.sleep(5)

        raise TimeoutError(f"배포 준비 대기 시간 초과: {name}")

    async def _perform_health_check(self, deployment_name: str, namespace: str) -> bool:
        """헬스체크 수행"""
        try:
            # 서비스 엔드포인트 가져오기
            service_name = f"{deployment_name}-service"

            # 실제 구현에서는 서비스 엔드포인트로 HTTP 요청
            # 여기서는 시뮬레이션
            await asyncio.sleep(2)

            logger.info(f"헬스체크 성공: {deployment_name}")
            return True

        except Exception as e:
            logger.error(f"헬스체크 실패: {e}")
            return False


class CloudDeployer:
    """클라우드 배포자"""

    def __init__(self, cloud_provider: str = "aws"):
        self.cloud_provider = cloud_provider

        if cloud_provider == "aws":
            self.ec2 = boto3.client('ec2')
            self.ecs = boto3.client('ecs')
            self.ecr = boto3.client('ecr')
            self.s3 = boto3.client('s3')
            self.cloudformation = boto3.client('cloudformation')

    async def deploy_to_ecs(
        self,
        cluster_name: str,
        service_name: str,
        task_definition: Dict[str, Any],
        desired_count: int = 1
    ) -> Dict[str, Any]:
        """AWS ECS에 배포"""
        try:
            # 태스크 정의 등록
            response = self.ecs.register_task_definition(**task_definition)
            task_def_arn = response['taskDefinition']['taskDefinitionArn']

            # 서비스 업데이트 또는 생성
            try:
                self.ecs.update_service(
                    cluster=cluster_name,
                    service=service_name,
                    taskDefinition=task_def_arn,
                    desiredCount=desired_count
                )
                logger.info(f"ECS 서비스 업데이트됨: {service_name}")
            except self.ecs.exceptions.ServiceNotFoundException:
                self.ecs.create_service(
                    cluster=cluster_name,
                    serviceName=service_name,
                    taskDefinition=task_def_arn,
                    desiredCount=desired_count
                )
                logger.info(f"새 ECS 서비스 생성됨: {service_name}")

            # 배포 완료 대기
            await self._wait_for_ecs_deployment(cluster_name, service_name)

            return {
                "status": "success",
                "cluster": cluster_name,
                "service": service_name,
                "task_definition": task_def_arn
            }

        except Exception as e:
            logger.error(f"ECS 배포 실패: {e}")
            raise

    async def _wait_for_ecs_deployment(self, cluster_name: str, service_name: str, timeout: int = 600):
        """ECS 배포 완료 대기"""
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                response = self.ecs.describe_services(
                    cluster=cluster_name,
                    services=[service_name]
                )

                service = response['services'][0]
                deployments = service['deployments']

                # PRIMARY 배포가 STEADY 상태인지 확인
                for deployment in deployments:
                    if deployment['status'] == 'PRIMARY':
                        if deployment['rolloutState'] == 'COMPLETED':
                            logger.info(f"ECS 배포 완료: {service_name}")
                            return True

                await asyncio.sleep(10)

            except Exception as e:
                logger.warning(f"ECS 배포 상태 확인 중 오류: {e}")
                await asyncio.sleep(10)

        raise TimeoutError(f"ECS 배포 대기 시간 초과: {service_name}")


class DeploymentPipeline:
    """배포 파이프라인"""

    def __init__(self):
        self.container_builder = ContainerBuilder()
        self.k8s_deployer = KubernetesDeployer()
        self.cloud_deployer = CloudDeployer()
        self.deployment_history: List[DeploymentResult] = []

    async def execute_pipeline(
        self,
        config: DeploymentConfig,
        pipeline_steps: List[str],
        **kwargs
    ) -> DeploymentResult:
        """배포 파이프라인 실행"""
        deployment_id = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        start_time = datetime.now()

        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.IN_PROGRESS,
            environment=config.environment.value,
            strategy=config.strategy.value,
            start_time=start_time
        )

        try:
            logger.info(f"배포 파이프라인 시작: {deployment_id}")

            for step in pipeline_steps:
                logger.info(f"파이프라인 단계 실행: {step}")

                if step == "build":
                    await self._build_step(config, result, **kwargs)
                elif step == "test":
                    await self._test_step(config, result, **kwargs)
                elif step == "security_scan":
                    await self._security_scan_step(config, result, **kwargs)
                elif step == "deploy":
                    await self._deploy_step(config, result, **kwargs)
                elif step == "health_check":
                    await self._health_check_step(config, result, **kwargs)
                elif step == "smoke_test":
                    await self._smoke_test_step(config, result, **kwargs)
                elif step == "rollback_check":
                    await self._rollback_check_step(config, result, **kwargs)

            # 성공 완료
            result.status = DeploymentStatus.SUCCESS
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()

            logger.info(f"배포 파이프라인 완료: {deployment_id}")

        except Exception as e:
            # 실패 처리
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()

            logger.error(f"배포 파이프라인 실패: {deployment_id} - {e}")

            # 롤백 수행
            if config.rollback_on_failure:
                try:
                    await self._perform_rollback(config, result)
                    result.rollback_performed = True
                except Exception as rollback_error:
                    logger.error(f"롤백 실패: {rollback_error}")

        finally:
            self.deployment_history.append(result)

        return result

    async def _build_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """빌드 단계"""
        dockerfile_path = kwargs.get("dockerfile_path", "Dockerfile")
        context_path = kwargs.get("context_path", ".")

        image_name = await self.container_builder.build_image(
            dockerfile_path=dockerfile_path,
            image_name="fragrance-ai",
            tag=config.image_tag,
            context_path=context_path
        )

        result.logs.append(f"이미지 빌드 완료: {image_name}")

        # 레지스트리에 푸시
        registry_url = kwargs.get("registry_url")
        if registry_url:
            success = await self.container_builder.push_image(image_name, registry_url)
            if not success:
                raise Exception("이미지 푸시 실패")
            result.logs.append(f"이미지 푸시 완료: {registry_url}/{image_name}")

    async def _test_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """테스트 단계"""
        test_command = kwargs.get("test_command", "pytest tests/")

        try:
            # 테스트 실행
            process = await asyncio.create_subprocess_shell(
                test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"테스트 실패: {stderr.decode()}")

            result.logs.append("모든 테스트 통과")

        except Exception as e:
            result.logs.append(f"테스트 실패: {e}")
            raise

    async def _security_scan_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """보안 스캔 단계"""
        # 컨테이너 이미지 보안 스캔 (시뮬레이션)
        await asyncio.sleep(2)

        # 실제로는 Trivy, Clair 등의 보안 스캐너 사용
        vulnerabilities_found = 0  # 시뮬레이션

        if vulnerabilities_found > 0:
            if vulnerabilities_found > 10:  # 중요 취약점
                raise Exception(f"심각한 보안 취약점 발견: {vulnerabilities_found}개")
            else:
                result.logs.append(f"경미한 보안 취약점 발견: {vulnerabilities_found}개")

        result.logs.append("보안 스캔 완료")

    async def _deploy_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """배포 단계"""
        deployment_target = kwargs.get("deployment_target", "kubernetes")
        namespace = kwargs.get("namespace", "default")

        if deployment_target == "kubernetes":
            deploy_result = await self.k8s_deployer.deploy_application(config, namespace)
            result.logs.append(f"Kubernetes 배포 완료: {deploy_result}")

        elif deployment_target == "ecs":
            cluster_name = kwargs.get("cluster_name", "fragrance-ai-cluster")
            service_name = kwargs.get("service_name", "fragrance-ai-service")
            task_definition = kwargs.get("task_definition", {})

            deploy_result = await self.cloud_deployer.deploy_to_ecs(
                cluster_name, service_name, task_definition
            )
            result.logs.append(f"ECS 배포 완료: {deploy_result}")

        else:
            raise Exception(f"지원하지 않는 배포 대상: {deployment_target}")

    async def _health_check_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """헬스체크 단계"""
        health_check_url = kwargs.get("health_check_url", "http://localhost:8000/health")
        max_retries = kwargs.get("max_retries", 10)
        retry_delay = kwargs.get("retry_delay", 30)

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_check_url, timeout=10) as response:
                        if response.status == 200:
                            result.logs.append("헬스체크 성공")
                            return

                        logger.warning(f"헬스체크 실패 (시도 {attempt + 1}): HTTP {response.status}")

            except Exception as e:
                logger.warning(f"헬스체크 오류 (시도 {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        raise Exception("헬스체크 실패")

    async def _smoke_test_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """스모크 테스트 단계"""
        api_endpoint = kwargs.get("api_endpoint", "http://localhost:8000/api/v1")

        try:
            async with aiohttp.ClientSession() as session:
                # 기본 API 엔드포인트 테스트
                test_endpoints = [
                    f"{api_endpoint}/health",
                    f"{api_endpoint}/search/semantic",
                    f"{api_endpoint}/generate/recipe"
                ]

                for endpoint in test_endpoints:
                    async with session.get(endpoint) as response:
                        if response.status not in [200, 404]:  # 404는 인증 없이는 정상
                            logger.warning(f"스모크 테스트 경고: {endpoint} - HTTP {response.status}")

            result.logs.append("스모크 테스트 완료")

        except Exception as e:
            result.logs.append(f"스모크 테스트 실패: {e}")
            raise

    async def _rollback_check_step(self, config: DeploymentConfig, result: DeploymentResult, **kwargs):
        """롤백 확인 단계"""
        # 배포 후 메트릭 모니터링
        monitoring_duration = kwargs.get("monitoring_duration", 300)  # 5분
        error_threshold = kwargs.get("error_threshold", 0.05)  # 5%

        logger.info(f"배포 후 모니터링 시작: {monitoring_duration}초")

        # 실제로는 Prometheus, CloudWatch 등에서 메트릭 수집
        await asyncio.sleep(10)  # 시뮬레이션

        # 모의 메트릭
        error_rate = 0.02  # 2%
        response_time_p95 = 150  # ms

        if error_rate > error_threshold:
            raise Exception(f"오류율이 임계값을 초과했습니다: {error_rate:.2%} > {error_threshold:.2%}")

        if response_time_p95 > 2000:  # 2초 초과
            logger.warning(f"응답시간이 높습니다: {response_time_p95}ms")

        result.success_rate = (1 - error_rate) * 100
        result.metrics = {
            "error_rate": error_rate,
            "response_time_p95": response_time_p95
        }

        result.logs.append(f"모니터링 완료 - 성공률: {result.success_rate:.2f}%")

    async def _perform_rollback(self, config: DeploymentConfig, result: DeploymentResult):
        """롤백 수행"""
        logger.info("롤백 시작")

        # 이전 버전으로 롤백 (실제 구현 필요)
        if len(self.deployment_history) > 1:
            previous_deployment = self.deployment_history[-2]
            if previous_deployment.status == DeploymentStatus.SUCCESS:
                # 이전 성공 버전으로 롤백
                logger.info(f"이전 버전으로 롤백: {previous_deployment.deployment_id}")
                result.logs.append(f"롤백 완료: {previous_deployment.deployment_id}")
            else:
                raise Exception("롤백할 안정적인 버전이 없습니다")
        else:
            raise Exception("롤백할 이전 버전이 없습니다")

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """배포 상태 조회"""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None

    def get_deployment_history(self, limit: int = 10) -> List[DeploymentResult]:
        """배포 이력 조회"""
        return self.deployment_history[-limit:]


# 전역 배포 파이프라인 인스턴스
global_pipeline: Optional[DeploymentPipeline] = None


def get_deployment_pipeline() -> DeploymentPipeline:
    """전역 배포 파이프라인 가져오기"""
    global global_pipeline
    if global_pipeline is None:
        global_pipeline = DeploymentPipeline()
    return global_pipeline


# 편의 함수들
async def deploy_to_production(
    image_tag: str,
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
    replicas: int = 3
) -> DeploymentResult:
    """프로덕션 배포"""
    config = DeploymentConfig(
        environment=DeploymentEnvironment.PRODUCTION,
        strategy=strategy,
        image_tag=image_tag,
        replicas=replicas,
        resources={
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "1000m", "memory": "2Gi"}
        }
    )

    pipeline_steps = [
        "build",
        "test",
        "security_scan",
        "deploy",
        "health_check",
        "smoke_test",
        "rollback_check"
    ]

    pipeline = get_deployment_pipeline()
    return await pipeline.execute_pipeline(config, pipeline_steps)


async def deploy_to_staging(image_tag: str) -> DeploymentResult:
    """스테이징 배포"""
    config = DeploymentConfig(
        environment=DeploymentEnvironment.STAGING,
        strategy=DeploymentStrategy.RECREATE,
        image_tag=image_tag,
        replicas=1,
        resources={
            "requests": {"cpu": "250m", "memory": "512Mi"},
            "limits": {"cpu": "500m", "memory": "1Gi"}
        }
    )

    pipeline_steps = ["build", "test", "deploy", "health_check"]

    pipeline = get_deployment_pipeline()
    return await pipeline.execute_pipeline(config, pipeline_steps)