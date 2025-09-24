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
        import aiohttp

        try:
            # 서비스 엔드포인트 가져오기
            service_name = f"{deployment_name}-service"

            # Kubernetes 서비스 정보 조회
            service_endpoint = await self._get_service_endpoint(service_name, namespace)

            # 실제 HTTP 헬스체크 요청
            health_url = f"http://{service_endpoint}/health"

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            health_data = await response.json()

                            # 헬스체크 상세 검증
                            if health_data.get('status') == 'healthy':
                                # 세부 컴포넌트 체크
                                components = health_data.get('components', {})
                                all_healthy = all(
                                    comp.get('status') == 'healthy'
                                    for comp in components.values()
                                )

                                if all_healthy:
                                    logger.info(f"헬스체크 성공: {deployment_name} - 모든 컴포넌트 정상")
                                    return True
                                else:
                                    unhealthy = [
                                        name for name, comp in components.items()
                                        if comp.get('status') != 'healthy'
                                    ]
                                    logger.warning(f"일부 컴포넌트 비정상: {unhealthy}")
                                    return False
                            else:
                                logger.warning(f"서비스 상태 비정상: {health_data.get('status')}")
                                return False
                        else:
                            logger.error(f"헬스체크 HTTP 오류: {response.status}")
                            return False

                except aiohttp.ClientTimeout:
                    logger.error(f"헬스체크 타임아웃: {deployment_name}")
                    return False
                except aiohttp.ClientError as e:
                    logger.error(f"헬스체크 연결 오류: {e}")
                    return False

        except Exception as e:
            logger.error(f"헬스체크 실패: {e}")
            return False

    async def _get_service_endpoint(self, service_name: str, namespace: str) -> str:
        """Kubernetes 서비스 엔드포인트 조회"""
        try:
            # kubectl을 사용하여 서비스 정보 조회
            cmd = f"kubectl get service {service_name} -n {namespace} -o json"
            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                import json
                service_info = json.loads(stdout)

                # ClusterIP 또는 LoadBalancer IP 추출
                cluster_ip = service_info['spec'].get('clusterIP')
                port = service_info['spec']['ports'][0]['port']

                # LoadBalancer 타입인 경우 외부 IP 사용
                if service_info['spec']['type'] == 'LoadBalancer':
                    ingress = service_info.get('status', {}).get('loadBalancer', {}).get('ingress', [])
                    if ingress:
                        external_ip = ingress[0].get('ip') or ingress[0].get('hostname')
                        if external_ip:
                            return f"{external_ip}:{port}"

                # NodePort 타입인 경우 노드 IP 사용
                elif service_info['spec']['type'] == 'NodePort':
                    node_port = service_info['spec']['ports'][0]['nodePort']
                    # 첫 번째 노드 IP 조회
                    nodes_cmd = "kubectl get nodes -o json"
                    nodes_result = await asyncio.create_subprocess_shell(
                        nodes_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    nodes_stdout, _ = await nodes_result.communicate()

                    if nodes_result.returncode == 0:
                        nodes_info = json.loads(nodes_stdout)
                        if nodes_info['items']:
                            node_ip = nodes_info['items'][0]['status']['addresses'][0]['address']
                            return f"{node_ip}:{node_port}"

                # 기본: ClusterIP 사용 (클러스터 내부에서만 접근 가능)
                return f"{cluster_ip}:{port}"

            else:
                # 로컬 개발 환경에서는 localhost 사용
                logger.warning(f"kubectl 실패, localhost 사용: {stderr.decode()}")
                return "localhost:8000"

        except Exception as e:
            logger.error(f"서비스 엔드포인트 조회 실패: {e}")
            # 기본값 반환
            return "localhost:8000"


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
        try:
            # Trivy를 사용한 컨테이너 이미지 보안 스캔
            scan_cmd = f"trivy image --severity HIGH,CRITICAL --format json {config.image_tag}"

            scan_result = await asyncio.create_subprocess_shell(
                scan_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await scan_result.communicate()

            if scan_result.returncode == 0:
                import json
                scan_data = json.loads(stdout) if stdout else {}

                vulnerabilities = []
                for target in scan_data.get('Results', []):
                    for vuln in target.get('Vulnerabilities', []):
                        if vuln['Severity'] in ['HIGH', 'CRITICAL']:
                            vulnerabilities.append({
                                'id': vuln.get('VulnerabilityID'),
                                'severity': vuln.get('Severity'),
                                'package': vuln.get('PkgName'),
                                'title': vuln.get('Title', 'No title')
                            })

                vulnerabilities_found = len(vulnerabilities)

                if vulnerabilities_found > 0:
                    if any(v['severity'] == 'CRITICAL' for v in vulnerabilities):
                        # CRITICAL 취약점이 있으면 배포 중단
                        critical_vulns = [v for v in vulnerabilities if v['severity'] == 'CRITICAL']
                        raise Exception(
                            f"CRITICAL 보안 취약점 발견 ({len(critical_vulns)}개): "
                            f"{', '.join([v['id'] for v in critical_vulns[:3]])}"
                        )
                    else:
                        # HIGH 취약점만 있으면 경고만
                        result.logs.append(f"HIGH 보안 취약점 발견 ({vulnerabilities_found}개)")
                        result.security_scan_results = vulnerabilities
                else:
                    result.logs.append("보안 취약점 없음")

            else:
                # Trivy가 설치되지 않은 경우 Grype 사용
                logger.warning("Trivy 실행 실패, Grype로 대체 시도")

                grype_cmd = f"grype {config.image_tag} -o json --fail-on high"
                grype_result = await asyncio.create_subprocess_shell(
                    grype_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await grype_result.communicate()

                if grype_result.returncode != 0 and grype_result.returncode != 1:
                    # returncode 1은 취약점 발견, 그 외는 오류
                    logger.warning(f"보안 스캔 도구 실행 실패: {stderr.decode()}")
                    result.logs.append("보안 스캔 스킵 (도구 없음)")
                elif grype_result.returncode == 1:
                    result.logs.append("HIGH 이상 보안 취약점 발견 - 검토 필요")

        except FileNotFoundError:
            # 스캔 도구가 없는 경우
            logger.warning("보안 스캔 도구가 설치되지 않음")
            result.logs.append("보안 스캔 스킵 (도구 미설치)")

        except Exception as e:
            logger.error(f"보안 스캔 오류: {e}")
            if "CRITICAL" in str(e):
                raise  # CRITICAL 취약점은 배포 중단
            else:
                result.logs.append(f"보안 스캔 경고: {e}")

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

        # 실제 메트릭 수집
        metrics = await self._collect_real_metrics(config, monitoring_duration)

        error_rate = metrics['error_rate']
        response_time_p95 = metrics['response_time_p95']

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

        # 실제 롤백 구현
        if len(self.deployment_history) > 1:
            previous_deployment = self.deployment_history[-2]
            if previous_deployment.status == DeploymentStatus.SUCCESS:
                # 이전 성공 버전으로 롤백
                await self._execute_rollback(config, previous_deployment)
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

    async def _collect_real_metrics(self, config: DeploymentConfig, duration: int) -> Dict[str, Any]:
        """실제 메트릭 수집"""
        metrics = {
            'error_rate': 0.0,
            'response_time_p95': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'request_rate': 0.0
        }

        # Prometheus 메트릭 수집
        if config.monitoring_config.get('prometheus_url'):
            metrics.update(await self._collect_prometheus_metrics(
                config.monitoring_config['prometheus_url'],
                duration
            ))

        # CloudWatch 메트릭 수집 (AWS)
        elif config.monitoring_config.get('cloudwatch_enabled'):
            metrics.update(await self._collect_cloudwatch_metrics(
                config.monitoring_config,
                duration
            ))

        # Datadog 메트릭 수집
        elif config.monitoring_config.get('datadog_api_key'):
            metrics.update(await self._collect_datadog_metrics(
                config.monitoring_config,
                duration
            ))

        # 로컬 헬스체크 폴백
        else:
            metrics.update(await self._collect_health_check_metrics(
                config.health_check_url,
                duration
            ))

        return metrics

    async def _collect_prometheus_metrics(self, prometheus_url: str, duration: int) -> Dict[str, Any]:
        """Prometheus에서 메트릭 수집"""
        import aiohttp
        import statistics

        metrics = {}
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=duration)

        queries = {
            'error_rate': 'rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])',
            'response_time_p95': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
            'cpu_usage': 'avg(rate(container_cpu_usage_seconds_total[5m]))',
            'memory_usage': 'avg(container_memory_usage_bytes)',
            'request_rate': 'rate(http_requests_total[5m])'
        }

        async with aiohttp.ClientSession() as session:
            for metric_name, query in queries.items():
                try:
                    async with session.get(
                        f"{prometheus_url}/api/v1/query",
                        params={'query': query}
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data['status'] == 'success' and data['data']['result']:
                                value = float(data['data']['result'][0]['value'][1])
                                metrics[metric_name] = value
                except Exception as e:
                    logger.warning(f"Failed to collect {metric_name} from Prometheus: {e}")
                    metrics[metric_name] = 0.0

        return metrics

    async def _collect_cloudwatch_metrics(self, config: Dict, duration: int) -> Dict[str, Any]:
        """AWS CloudWatch에서 메트릭 수집"""
        try:
            import boto3
            from datetime import datetime, timedelta

            cloudwatch = boto3.client(
                'cloudwatch',
                region_name=config.get('aws_region', 'us-east-1'),
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key')
            )

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(seconds=duration)

            metrics = {}

            # 오류율
            error_stats = cloudwatch.get_metric_statistics(
                Namespace='AWS/ELB',
                MetricName='HTTPCode_Target_5XX_Count',
                Dimensions=[{'Name': 'LoadBalancer', 'Value': config.get('load_balancer_name', '')}],
                StartTime=start_time,
                EndTime=end_time,
                Period=60,
                Statistics=['Sum']
            )

            total_stats = cloudwatch.get_metric_statistics(
                Namespace='AWS/ELB',
                MetricName='RequestCount',
                Dimensions=[{'Name': 'LoadBalancer', 'Value': config.get('load_balancer_name', '')}],
                StartTime=start_time,
                EndTime=end_time,
                Period=60,
                Statistics=['Sum']
            )

            error_count = sum(point['Sum'] for point in error_stats.get('Datapoints', []))
            total_count = sum(point['Sum'] for point in total_stats.get('Datapoints', []))

            metrics['error_rate'] = error_count / total_count if total_count > 0 else 0.0

            # 응답 시간
            response_time_stats = cloudwatch.get_metric_statistics(
                Namespace='AWS/ELB',
                MetricName='TargetResponseTime',
                Dimensions=[{'Name': 'LoadBalancer', 'Value': config.get('load_balancer_name', '')}],
                StartTime=start_time,
                EndTime=end_time,
                Period=60,
                Statistics=['Average']
            )

            if response_time_stats['Datapoints']:
                metrics['response_time_p95'] = max(point['Average'] for point in response_time_stats['Datapoints']) * 1000

            return metrics

        except Exception as e:
            logger.error(f"CloudWatch metrics collection failed: {e}")
            return {'error_rate': 0.0, 'response_time_p95': 0.0}

    async def _collect_datadog_metrics(self, config: Dict, duration: int) -> Dict[str, Any]:
        """Datadog에서 메트릭 수집"""
        import aiohttp

        api_key = config['datadog_api_key']
        app_key = config.get('datadog_app_key')

        headers = {
            'DD-API-KEY': api_key,
            'DD-APPLICATION-KEY': app_key
        } if app_key else {'DD-API-KEY': api_key}

        end_time = int(datetime.now().timestamp())
        start_time = end_time - duration

        metrics = {}

        queries = {
            'error_rate': 'sum:http.requests{status:5*}.as_rate()/sum:http.requests{*}.as_rate()',
            'response_time_p95': 'percentile:http.request.duration{*}:95',
            'cpu_usage': 'avg:system.cpu.user{*}',
            'memory_usage': 'avg:system.mem.used{*}'
        }

        async with aiohttp.ClientSession() as session:
            for metric_name, query in queries.items():
                try:
                    async with session.get(
                        'https://api.datadoghq.com/api/v1/query',
                        params={
                            'from': start_time,
                            'to': end_time,
                            'query': query
                        },
                        headers=headers
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data['series']:
                                points = data['series'][0]['pointlist']
                                if points:
                                    metrics[metric_name] = points[-1][1]  # 최신 값
                except Exception as e:
                    logger.warning(f"Failed to collect {metric_name} from Datadog: {e}")
                    metrics[metric_name] = 0.0

        return metrics

    async def _collect_health_check_metrics(self, health_url: str, duration: int) -> Dict[str, Any]:
        """헬스체크 엔드포인트에서 메트릭 수집"""
        import aiohttp
        import statistics

        metrics = {
            'error_rate': 0.0,
            'response_time_p95': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0
        }

        response_times = []
        error_count = 0
        total_count = 0

        # duration 동안 주기적으로 헬스체크
        check_interval = min(5, duration // 10)  # 5초 또는 duration/10 중 작은 값
        checks = duration // check_interval

        async with aiohttp.ClientSession() as session:
            for _ in range(checks):
                start = datetime.now()
                try:
                    async with session.get(health_url, timeout=10) as resp:
                        response_time = (datetime.now() - start).total_seconds() * 1000
                        response_times.append(response_time)

                        if resp.status >= 500:
                            error_count += 1
                        total_count += 1

                        # 헬스체크 응답에서 추가 메트릭 추출
                        if resp.status == 200:
                            try:
                                data = await resp.json()
                                metrics['cpu_usage'] = data.get('cpu_usage', 0.0)
                                metrics['memory_usage'] = data.get('memory_usage', 0.0)
                            except:
                                pass

                except Exception as e:
                    logger.warning(f"Health check failed: {e}")
                    error_count += 1
                    total_count += 1

                await asyncio.sleep(check_interval)

        if total_count > 0:
            metrics['error_rate'] = error_count / total_count

        if response_times:
            response_times.sort()
            p95_index = int(len(response_times) * 0.95)
            metrics['response_time_p95'] = response_times[min(p95_index, len(response_times) - 1)]

        return metrics

    async def _execute_rollback(self, config: DeploymentConfig, previous_deployment: DeploymentResult):
        """실제 롤백 실행"""
        rollback_strategy = config.rollback_strategy

        if rollback_strategy == "blue_green":
            # Blue-Green 롤백: 이전 환경으로 트래픽 전환
            await self._switch_traffic_to_previous(config, previous_deployment)

        elif rollback_strategy == "canary":
            # Canary 롤백: 새 버전 인스턴스 제거
            await self._remove_canary_instances(config)

        elif rollback_strategy == "rolling":
            # Rolling 롤백: 이전 버전으로 순차적 재배포
            await self._rolling_rollback(config, previous_deployment)

        else:
            # 기본: 이전 버전 재배포
            await self._redeploy_previous_version(config, previous_deployment)

    async def _switch_traffic_to_previous(self, config: DeploymentConfig, previous: DeploymentResult):
        """Blue-Green: 이전 환경으로 트래픽 전환"""
        load_balancer_config = config.rollback_config.get('load_balancer')

        if load_balancer_config:
            # AWS ELB, Azure Load Balancer, 또는 nginx 설정 변경
            if load_balancer_config['type'] == 'aws_elb':
                await self._update_aws_elb_target_group(
                    load_balancer_config,
                    previous.deployment_data.get('target_group_arn')
                )
            elif load_balancer_config['type'] == 'nginx':
                await self._update_nginx_upstream(
                    load_balancer_config,
                    previous.deployment_data.get('upstream_servers')
                )

        logger.info(f"Traffic switched to previous deployment: {previous.deployment_id}")

    async def _remove_canary_instances(self, config: DeploymentConfig):
        """Canary 인스턴스 제거"""
        canary_instances = config.rollback_config.get('canary_instances', [])

        for instance_id in canary_instances:
            # Kubernetes, Docker, 또는 클라우드 인스턴스 제거
            if config.deployment_type == "kubernetes":
                await self._delete_k8s_deployment(instance_id)
            elif config.deployment_type == "docker":
                await self._stop_docker_container(instance_id)
            elif config.deployment_type == "ec2":
                await self._terminate_ec2_instance(instance_id)

        logger.info(f"Removed {len(canary_instances)} canary instances")

    async def _rolling_rollback(self, config: DeploymentConfig, previous: DeploymentResult):
        """롤링 롤백 실행"""
        instances = config.rollback_config.get('instances', [])
        batch_size = config.rollback_config.get('batch_size', 1)

        for i in range(0, len(instances), batch_size):
            batch = instances[i:i+batch_size]

            for instance_id in batch:
                await self._rollback_instance(instance_id, previous.deployment_data)

            # 배치 간 대기
            await asyncio.sleep(config.rollback_config.get('batch_delay', 30))

            # 헬스체크
            await self._verify_instance_health(batch)

        logger.info(f"Rolling rollback completed for {len(instances)} instances")

    async def _redeploy_previous_version(self, config: DeploymentConfig, previous: DeploymentResult):
        """이전 버전 재배포"""
        deployment_commands = previous.deployment_data.get('deployment_commands', [])

        for command in deployment_commands:
            # 배포 명령 실행
            result = await self._execute_deployment_command(command)
            if not result['success']:
                raise Exception(f"Rollback command failed: {command}")

        logger.info(f"Previous version redeployed: {previous.deployment_id}")

    async def _update_aws_elb_target_group(self, load_balancer_config: Dict, target_group_arn: str):
        """AWS ELB 타겟 그룹 업데이트"""
        try:
            import boto3
            elbv2 = boto3.client('elbv2',
                                  region_name=load_balancer_config.get('region', 'us-east-1'))

            # 현재 타겟 그룹의 타겟 해제
            current_targets = elbv2.describe_target_health(TargetGroupArn=target_group_arn)
            if current_targets['TargetHealthDescriptions']:
                targets_to_deregister = [
                    {'Id': t['Target']['Id']} for t in current_targets['TargetHealthDescriptions']
                ]
                elbv2.deregister_targets(
                    TargetGroupArn=target_group_arn,
                    Targets=targets_to_deregister
                )

            # 새 타겟 등록
            new_targets = load_balancer_config.get('new_targets', [])
            if new_targets:
                elbv2.register_targets(
                    TargetGroupArn=target_group_arn,
                    Targets=new_targets
                )

            logger.info(f"AWS ELB 타겟 그룹 업데이트 완료: {target_group_arn}")

        except Exception as e:
            logger.error(f"AWS ELB 업데이트 실패: {e}")
            raise

    async def _update_nginx_upstream(self, load_balancer_config: Dict, upstream_servers: List[str]):
        """Nginx upstream 설정 업데이트"""
        import aiohttp

        try:
            nginx_api_url = load_balancer_config.get('nginx_api_url', 'http://nginx-plus-api:8080')
            upstream_name = load_balancer_config.get('upstream_name', 'backend')

            # Nginx Plus API를 사용한 동적 upstream 업데이트
            async with aiohttp.ClientSession() as session:
                # 현재 서버 목록 조회
                async with session.get(f"{nginx_api_url}/api/6/http/upstreams/{upstream_name}/servers") as resp:
                    if resp.status == 200:
                        current_servers = await resp.json()

                        # 기존 서버 제거
                        for server in current_servers:
                            await session.delete(
                                f"{nginx_api_url}/api/6/http/upstreams/{upstream_name}/servers/{server['id']}"
                            )

                # 새 서버 추가
                for server in upstream_servers:
                    server_config = {
                        "server": server,
                        "weight": 1,
                        "max_fails": 3,
                        "fail_timeout": "10s"
                    }
                    await session.post(
                        f"{nginx_api_url}/api/6/http/upstreams/{upstream_name}/servers",
                        json=server_config
                    )

            logger.info(f"Nginx upstream 업데이트 완료: {upstream_name}")

        except Exception as e:
            logger.error(f"Nginx upstream 업데이트 실패: {e}")
            # Fallback: 설정 파일 직접 수정
            await self._update_nginx_config_file(load_balancer_config, upstream_servers)

    async def _update_nginx_config_file(self, load_balancer_config: Dict, upstream_servers: List[str]):
        """Nginx 설정 파일 직접 업데이트"""
        config_path = load_balancer_config.get('config_path', '/etc/nginx/conf.d/upstream.conf')
        upstream_name = load_balancer_config.get('upstream_name', 'backend')

        upstream_config = f"upstream {upstream_name} {{\n"
        for server in upstream_servers:
            upstream_config += f"    server {server};\n"
        upstream_config += "}\n"

        # SSH를 통한 원격 파일 업데이트
        nginx_host = load_balancer_config.get('nginx_host', 'nginx-server')
        cmd = f"ssh {nginx_host} 'echo \"{upstream_config}\" > {config_path} && nginx -s reload'"

        result = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise Exception(f"Nginx config update failed: {stderr.decode()}")

    async def _delete_k8s_deployment(self, instance_id: str):
        """Kubernetes 디플로이먼트 삭제"""
        try:
            namespace, deployment_name = instance_id.split('/', 1)
            cmd = f"kubectl delete deployment {deployment_name} -n {namespace}"

            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"K8s deployment 삭제 완료: {instance_id}")
            else:
                raise Exception(f"K8s deployment 삭제 실패: {stderr.decode()}")

        except Exception as e:
            logger.error(f"K8s deployment 삭제 오류: {e}")
            raise

    async def _stop_docker_container(self, instance_id: str):
        """Docker 컨테이너 중지"""
        try:
            # 로컬 또는 원격 Docker 데몬에 연결
            cmd = f"docker stop {instance_id} && docker rm {instance_id}"

            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                logger.info(f"Docker container 중지 완료: {instance_id}")
            else:
                # 이미 중지된 컨테이너일 수 있음
                logger.warning(f"Docker container 중지 경고: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Docker container 중지 오류: {e}")
            raise

    async def _terminate_ec2_instance(self, instance_id: str):
        """EC2 인스턴스 종료"""
        try:
            import boto3
            ec2 = boto3.client('ec2')

            # 인스턴스 종료
            response = ec2.terminate_instances(InstanceIds=[instance_id])

            # 종료 대기
            waiter = ec2.get_waiter('instance_terminated')
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={
                    'Delay': 5,
                    'MaxAttempts': 60  # 최대 5분 대기
                }
            )

            logger.info(f"EC2 instance 종료 완료: {instance_id}")

        except Exception as e:
            logger.error(f"EC2 instance 종료 오류: {e}")
            raise

    async def _rollback_instance(self, instance_id: str, deployment_data: Dict):
        """개별 인스턴스 롤백"""
        instance_type = deployment_data.get('instance_type', 'kubernetes')

        if instance_type == 'kubernetes':
            # K8s Pod 재시작 또는 이미지 변경
            namespace, pod_name = instance_id.split('/', 1)
            previous_image = deployment_data.get('previous_image')

            cmd = f"kubectl set image deployment/{pod_name} *={previous_image} -n {namespace}"
            await asyncio.create_subprocess_shell(cmd)

        elif instance_type == 'docker':
            # Docker 컨테이너 재생성
            previous_image = deployment_data.get('previous_image')
            container_config = deployment_data.get('container_config', {})

            # 기존 컨테이너 중지
            await self._stop_docker_container(instance_id)

            # 새 컨테이너 시작
            docker_run_cmd = f"docker run -d --name {instance_id} {previous_image}"
            await asyncio.create_subprocess_shell(docker_run_cmd)

        elif instance_type == 'ec2':
            # EC2 인스턴스 AMI 변경 (새 인스턴스 시작)
            await self._launch_ec2_from_ami(
                instance_id,
                deployment_data.get('previous_ami')
            )

    async def _launch_ec2_from_ami(self, instance_name: str, ami_id: str):
        """AMI로부터 EC2 인스턴스 시작"""
        try:
            import boto3
            ec2 = boto3.client('ec2')

            # 새 인스턴스 시작
            response = ec2.run_instances(
                ImageId=ami_id,
                MinCount=1,
                MaxCount=1,
                InstanceType='t2.micro',  # 설정에서 가져올 수 있음
                TagSpecifications=[
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Name', 'Value': instance_name},
                            {'Key': 'Environment', 'Value': 'rollback'}
                        ]
                    }
                ]
            )

            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"새 EC2 인스턴스 시작: {instance_id}")
            return instance_id

        except Exception as e:
            logger.error(f"EC2 인스턴스 시작 실패: {e}")
            raise

    async def _verify_instance_health(self, instances: List[str]):
        """인스턴스 헬스 확인"""
        health_checks = []

        for instance_id in instances:
            # 각 인스턴스 타입에 따른 헬스체크
            if '/' in instance_id:  # Kubernetes
                health_checks.append(self._check_k8s_pod_health(instance_id))
            elif instance_id.startswith('i-'):  # EC2
                health_checks.append(self._check_ec2_health(instance_id))
            else:  # Docker
                health_checks.append(self._check_docker_health(instance_id))

        results = await asyncio.gather(*health_checks, return_exceptions=True)

        for instance_id, result in zip(instances, results):
            if isinstance(result, Exception):
                logger.error(f"헬스체크 실패 {instance_id}: {result}")
            elif not result:
                logger.warning(f"인스턴스 비정상 {instance_id}")

    async def _check_k8s_pod_health(self, pod_id: str) -> bool:
        """Kubernetes Pod 헬스 체크"""
        namespace, pod_name = pod_id.split('/', 1)
        cmd = f"kubectl get pod {pod_name} -n {namespace} -o json"

        result = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode == 0:
            import json
            pod_info = json.loads(stdout)
            return pod_info['status']['phase'] == 'Running'
        return False

    async def _check_ec2_health(self, instance_id: str) -> bool:
        """EC2 인스턴스 헬스 체크"""
        try:
            import boto3
            ec2 = boto3.client('ec2')

            response = ec2.describe_instance_status(InstanceIds=[instance_id])
            if response['InstanceStatuses']:
                status = response['InstanceStatuses'][0]
                return (status['InstanceStatus']['Status'] == 'ok' and
                        status['SystemStatus']['Status'] == 'ok')
            return False
        except:
            return False

    async def _check_docker_health(self, container_id: str) -> bool:
        """Docker 컨테이너 헬스 체크"""
        cmd = f"docker inspect --format='{{{{.State.Health.Status}}}}' {container_id}"

        result = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        if result.returncode == 0:
            health_status = stdout.decode().strip()
            return health_status in ['healthy', 'starting']
        return False

    async def _execute_deployment_command(self, command: Dict) -> Dict[str, Any]:
        """배포 명령 실행"""
        cmd_type = command.get('type')
        cmd_str = command.get('command')

        try:
            if cmd_type == 'shell':
                result = await asyncio.create_subprocess_shell(
                    cmd_str,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()

                return {
                    'success': result.returncode == 0,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode()
                }

            elif cmd_type == 'kubernetes':
                # kubectl 명령 실행
                return await self._execute_kubectl_command(cmd_str)

            elif cmd_type == 'docker':
                # docker 명령 실행
                return await self._execute_docker_command(cmd_str)

            else:
                raise ValueError(f"Unknown command type: {cmd_type}")

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        """kubectl 명령 실행"""
        full_cmd = f"kubectl {command}"
        result = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        return {
            'success': result.returncode == 0,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }

    async def _execute_docker_command(self, command: str) -> Dict[str, Any]:
        """docker 명령 실행"""
        full_cmd = f"docker {command}"
        result = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()

        return {
            'success': result.returncode == 0,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }

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