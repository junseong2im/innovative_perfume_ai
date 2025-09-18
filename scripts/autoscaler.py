#!/usr/bin/env python3
"""
Docker Compose 기반 오토스케일러
CPU, 메모리 사용률 기반 자동 스케일링
"""

import docker
import time
import logging
import os
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScaleConfig:
    """스케일링 설정"""
    service_name: str
    min_replicas: int = 2
    max_replicas: int = 10
    scale_up_threshold: float = 80.0  # CPU %
    scale_down_threshold: float = 30.0  # CPU %
    scale_up_cooldown: int = 300  # 5분
    scale_down_cooldown: int = 600  # 10분
    check_interval: int = 30  # 30초

class AutoScaler:
    """Docker Compose 오토스케일러"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.scale_configs = {
            'fragrance_ai': ScaleConfig(
                service_name='fragrance_ai',
                min_replicas=int(os.getenv('MIN_REPLICAS', 2)),
                max_replicas=int(os.getenv('MAX_REPLICAS', 10)),
                scale_up_threshold=float(os.getenv('SCALE_UP_THRESHOLD', 80)),
                scale_down_threshold=float(os.getenv('SCALE_DOWN_THRESHOLD', 30)),
            ),
            'celery_worker': ScaleConfig(
                service_name='celery_worker',
                min_replicas=2,
                max_replicas=15,
                scale_up_threshold=70.0,
                scale_down_threshold=25.0,
            )
        }

        self.last_scale_actions = {}
        self.metrics_history = {}

    def get_service_containers(self, service_name: str) -> List[docker.models.containers.Container]:
        """서비스의 모든 컨테이너 조회"""
        try:
            containers = []
            all_containers = self.docker_client.containers.list(
                filters={'label': f'com.docker.compose.service={service_name}'}
            )

            for container in all_containers:
                if container.status == 'running':
                    containers.append(container)

            return containers

        except Exception as e:
            logger.error(f"Error getting containers for {service_name}: {e}")
            return []

    def get_container_metrics(self, container) -> Optional[Dict]:
        """컨테이너 메트릭 조회"""
        try:
            stats = container.stats(stream=False)

            # CPU 사용률 계산
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']

            cpu_percent = 0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             stats['cpu_stats']['online_cpus'] * 100

            # 메모리 사용률 계산
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting metrics for container {container.name}: {e}")
            return None

    def get_service_average_metrics(self, service_name: str) -> Optional[Dict]:
        """서비스의 평균 메트릭 계산"""
        containers = self.get_service_containers(service_name)
        if not containers:
            return None

        total_cpu = 0
        total_memory = 0
        valid_containers = 0

        for container in containers:
            metrics = self.get_container_metrics(container)
            if metrics:
                total_cpu += metrics['cpu_percent']
                total_memory += metrics['memory_percent']
                valid_containers += 1

        if valid_containers == 0:
            return None

        return {
            'avg_cpu_percent': total_cpu / valid_containers,
            'avg_memory_percent': total_memory / valid_containers,
            'container_count': valid_containers,
            'timestamp': datetime.now()
        }

    def should_scale_up(self, service_name: str, metrics: Dict, config: ScaleConfig) -> bool:
        """스케일 업 여부 결정"""

        # 최대 replica 확인
        if metrics['container_count'] >= config.max_replicas:
            return False

        # CPU 임계값 확인
        if metrics['avg_cpu_percent'] < config.scale_up_threshold:
            return False

        # 쿨다운 확인
        last_action_time = self.last_scale_actions.get(f"{service_name}_up")
        if last_action_time:
            if datetime.now() - last_action_time < timedelta(seconds=config.scale_up_cooldown):
                return False

        # 메트릭 히스토리 확인 (지속적인 높은 사용률)
        history_key = f"{service_name}_cpu"
        if history_key not in self.metrics_history:
            self.metrics_history[history_key] = []

        self.metrics_history[history_key].append(metrics['avg_cpu_percent'])

        # 최근 5개 메트릭의 평균이 임계값을 초과하는지 확인
        recent_metrics = self.metrics_history[history_key][-5:]
        if len(recent_metrics) >= 3:
            avg_recent_cpu = sum(recent_metrics) / len(recent_metrics)
            return avg_recent_cpu >= config.scale_up_threshold

        return False

    def should_scale_down(self, service_name: str, metrics: Dict, config: ScaleConfig) -> bool:
        """스케일 다운 여부 결정"""

        # 최소 replica 확인
        if metrics['container_count'] <= config.min_replicas:
            return False

        # CPU 임계값 확인
        if metrics['avg_cpu_percent'] > config.scale_down_threshold:
            return False

        # 쿨다운 확인
        last_action_time = self.last_scale_actions.get(f"{service_name}_down")
        if last_action_time:
            if datetime.now() - last_action_time < timedelta(seconds=config.scale_down_cooldown):
                return False

        # 메트릭 히스토리 확인 (지속적인 낮은 사용률)
        history_key = f"{service_name}_cpu"
        if history_key not in self.metrics_history:
            return False

        recent_metrics = self.metrics_history[history_key][-10:]  # 더 긴 기간 확인
        if len(recent_metrics) >= 5:
            avg_recent_cpu = sum(recent_metrics) / len(recent_metrics)
            return avg_recent_cpu <= config.scale_down_threshold

        return False

    def scale_service(self, service_name: str, target_replicas: int) -> bool:
        """서비스 스케일링 실행"""
        try:
            # Docker Compose 스케일 명령 실행
            import subprocess
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.scale.yml',
                'up', '-d', '--scale', f'{service_name}={target_replicas}'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully scaled {service_name} to {target_replicas} replicas")

                # HAProxy 설정 업데이트 (필요시)
                self.update_haproxy_config(service_name, target_replicas)

                return True
            else:
                logger.error(f"Failed to scale {service_name}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error scaling {service_name}: {e}")
            return False

    def update_haproxy_config(self, service_name: str, replicas: int):
        """HAProxy 설정 동적 업데이트"""
        try:
            # HAProxy Stats Socket을 통한 동적 서버 관리
            # 실제 구현에서는 HAProxy Management API 사용
            logger.info(f"HAProxy config update needed for {service_name} with {replicas} replicas")

        except Exception as e:
            logger.error(f"Error updating HAProxy config: {e}")

    def get_queue_length(self) -> int:
        """Celery 큐 길이 조회"""
        try:
            # Redis에서 Celery 큐 길이 확인
            import redis
            r = redis.Redis(host='redis', port=6379, password=os.getenv('REDIS_PASSWORD'), db=1)

            queue_length = r.llen('celery')  # 기본 큐
            return queue_length

        except Exception as e:
            logger.error(f"Error getting queue length: {e}")
            return 0

    def custom_celery_scaling_logic(self, service_name: str, config: ScaleConfig) -> Optional[int]:
        """Celery 워커 커스텀 스케일링 로직"""

        queue_length = self.get_queue_length()
        current_workers = len(self.get_service_containers(service_name))

        # 큐에 쌓인 작업 기준 스케일링
        tasks_per_worker = 10  # 워커당 처리할 수 있는 작업 수

        optimal_workers = min(
            max(
                config.min_replicas,
                (queue_length // tasks_per_worker) + 1
            ),
            config.max_replicas
        )

        if optimal_workers != current_workers:
            logger.info(f"Celery queue length: {queue_length}, current workers: {current_workers}, optimal: {optimal_workers}")
            return optimal_workers

        return None

    def run_scaling_check(self):
        """스케일링 검사 실행"""
        logger.info("Running scaling check...")

        for service_name, config in self.scale_configs.items():
            try:
                # 커스텀 로직 (Celery의 경우)
                if service_name == 'celery_worker':
                    target_replicas = self.custom_celery_scaling_logic(service_name, config)
                    if target_replicas:
                        if self.scale_service(service_name, target_replicas):
                            self.last_scale_actions[f"{service_name}_custom"] = datetime.now()
                        continue

                # 일반 메트릭 기반 스케일링
                metrics = self.get_service_average_metrics(service_name)
                if not metrics:
                    logger.warning(f"No metrics available for {service_name}")
                    continue

                current_replicas = metrics['container_count']
                logger.info(f"{service_name}: {current_replicas} replicas, "
                          f"CPU: {metrics['avg_cpu_percent']:.1f}%, "
                          f"Memory: {metrics['avg_memory_percent']:.1f}%")

                # 스케일 업 검사
                if self.should_scale_up(service_name, metrics, config):
                    new_replicas = min(current_replicas + 1, config.max_replicas)
                    logger.info(f"Scaling UP {service_name}: {current_replicas} -> {new_replicas}")

                    if self.scale_service(service_name, new_replicas):
                        self.last_scale_actions[f"{service_name}_up"] = datetime.now()

                # 스케일 다운 검사
                elif self.should_scale_down(service_name, metrics, config):
                    new_replicas = max(current_replicas - 1, config.min_replicas)
                    logger.info(f"Scaling DOWN {service_name}: {current_replicas} -> {new_replicas}")

                    if self.scale_service(service_name, new_replicas):
                        self.last_scale_actions[f"{service_name}_down"] = datetime.now()

            except Exception as e:
                logger.error(f"Error processing {service_name}: {e}")

    def run(self):
        """메인 실행 루프"""
        logger.info("Starting AutoScaler...")
        logger.info(f"Scale configs: {self.scale_configs}")

        while True:
            try:
                self.run_scaling_check()

                # 메트릭 히스토리 정리 (1시간 이상 된 데이터 삭제)
                cutoff_time = datetime.now() - timedelta(hours=1)
                for key in list(self.metrics_history.keys()):
                    self.metrics_history[key] = [
                        metric for metric in self.metrics_history[key]
                        if isinstance(metric, (int, float)) or
                        (hasattr(metric, 'timestamp') and metric.timestamp > cutoff_time)
                    ]

                # 대기
                check_interval = int(os.getenv('CHECK_INTERVAL', 30))
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("AutoScaler stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(60)  # 에러 시 1분 대기

if __name__ == "__main__":
    autoscaler = AutoScaler()
    autoscaler.run()