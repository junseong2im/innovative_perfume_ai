"""
모델 핫 리로드 (Model Hot Reload)

새 워커 warm-up → 트래픽 전환 (graceful reload)
"""

import torch
import threading
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 워커 상태
# =============================================================================

class WorkerStatus(Enum):
    """워커 상태"""
    INITIALIZING = "initializing"  # 초기화 중
    WARMING_UP = "warming_up"      # warm-up 중
    READY = "ready"                # 준비 완료
    SERVING = "serving"            # 서빙 중
    DRAINING = "draining"          # 요청 배출 중
    STOPPED = "stopped"            # 정지됨


@dataclass
class WorkerInfo:
    """워커 정보"""
    worker_id: str
    status: WorkerStatus
    model: Any
    in_flight_requests: int = 0
    total_requests_served: int = 0


# =============================================================================
# 모델 워커
# =============================================================================

class ModelWorker:
    """
    모델 워커

    모델을 로드하고 추론 요청을 처리합니다.
    """

    def __init__(
        self,
        worker_id: str,
        model_loader: Callable[[], Any],
        warmup_func: Optional[Callable[[Any], None]] = None,
        warmup_steps: int = 5
    ):
        """
        Args:
            worker_id: 워커 ID
            model_loader: 모델 로더 함수
            warmup_func: Warm-up 함수 (optional)
            warmup_steps: Warm-up 스텝 수
        """
        self.worker_id = worker_id
        self.model_loader = model_loader
        self.warmup_func = warmup_func
        self.warmup_steps = warmup_steps

        self.status = WorkerStatus.INITIALIZING
        self.model: Optional[Any] = None
        self.in_flight_requests = 0
        self.total_requests_served = 0
        self.lock = threading.Lock()

        logger.info(f"[Worker {worker_id}] 초기화 중")

    def initialize(self):
        """모델 초기화"""
        try:
            self.status = WorkerStatus.INITIALIZING
            logger.info(f"[Worker {self.worker_id}] 모델 로드 시작")

            # 모델 로드
            self.model = self.model_loader()

            logger.info(f"[Worker {self.worker_id}] 모델 로드 완료")

            # Warm-up
            if self.warmup_func and self.model:
                self.status = WorkerStatus.WARMING_UP
                logger.info(f"[Worker {self.worker_id}] Warm-up 시작 ({self.warmup_steps} steps)")

                for i in range(self.warmup_steps):
                    self.warmup_func(self.model)
                    logger.debug(f"[Worker {self.worker_id}] Warm-up step {i+1}/{self.warmup_steps}")

                logger.info(f"[Worker {self.worker_id}] Warm-up 완료")

            # 준비 완료
            self.status = WorkerStatus.READY
            logger.info(f"[Worker {self.worker_id}] 준비 완료")

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] 초기화 실패: {e}")
            self.status = WorkerStatus.STOPPED
            raise

    def serve(self, request_func: Callable[[Any], Any], *args, **kwargs) -> Any:
        """
        요청 처리

        Args:
            request_func: 요청 처리 함수
            *args, **kwargs: 요청 함수 인자

        Returns:
            요청 결과
        """
        if self.status not in [WorkerStatus.READY, WorkerStatus.SERVING]:
            raise RuntimeError(f"워커가 준비되지 않음: {self.status}")

        with self.lock:
            self.in_flight_requests += 1
            if self.status == WorkerStatus.READY:
                self.status = WorkerStatus.SERVING

        try:
            result = request_func(self.model, *args, **kwargs)
            self.total_requests_served += 1
            return result

        finally:
            with self.lock:
                self.in_flight_requests -= 1

    def drain(self, timeout: float = 30.0):
        """
        요청 배출 (진행 중인 요청 완료 대기)

        Args:
            timeout: 타임아웃 (초)
        """
        self.status = WorkerStatus.DRAINING
        logger.info(f"[Worker {self.worker_id}] 요청 배출 시작 (in-flight: {self.in_flight_requests})")

        start_time = time.time()
        while self.in_flight_requests > 0:
            if time.time() - start_time > timeout:
                logger.warning(f"[Worker {self.worker_id}] 배출 타임아웃 (남은 요청: {self.in_flight_requests})")
                break

            time.sleep(0.1)

        logger.info(f"[Worker {self.worker_id}] 요청 배출 완료")

    def stop(self):
        """워커 정지"""
        self.status = WorkerStatus.STOPPED
        self.model = None
        logger.info(f"[Worker {self.worker_id}] 정지됨")

    def get_info(self) -> WorkerInfo:
        """워커 정보 조회"""
        return WorkerInfo(
            worker_id=self.worker_id,
            status=self.status,
            model=self.model,
            in_flight_requests=self.in_flight_requests,
            total_requests_served=self.total_requests_served
        )


# =============================================================================
# 핫 리로드 매니저
# =============================================================================

class HotReloadManager:
    """
    핫 리로드 매니저

    새 워커를 warm-up하고 트래픽을 전환합니다 (graceful reload).
    """

    def __init__(
        self,
        model_loader: Callable[[], Any],
        warmup_func: Optional[Callable[[Any], None]] = None,
        warmup_steps: int = 5,
        drain_timeout: float = 30.0
    ):
        """
        Args:
            model_loader: 모델 로더 함수
            warmup_func: Warm-up 함수 (optional)
            warmup_steps: Warm-up 스텝 수
            drain_timeout: 배출 타임아웃 (초)
        """
        self.model_loader = model_loader
        self.warmup_func = warmup_func
        self.warmup_steps = warmup_steps
        self.drain_timeout = drain_timeout

        self.current_worker: Optional[ModelWorker] = None
        self.worker_counter = 0
        self.lock = threading.Lock()

        logger.info("핫 리로드 매니저 초기화")

    def initialize_first_worker(self):
        """첫 번째 워커 초기화"""
        with self.lock:
            if self.current_worker:
                logger.warning("이미 워커가 초기화되어 있음")
                return

            worker_id = f"worker-{self.worker_counter}"
            self.worker_counter += 1

            worker = ModelWorker(
                worker_id=worker_id,
                model_loader=self.model_loader,
                warmup_func=self.warmup_func,
                warmup_steps=self.warmup_steps
            )

            worker.initialize()
            self.current_worker = worker

            logger.info(f"첫 번째 워커 초기화 완료: {worker_id}")

    def reload(self) -> bool:
        """
        모델 리로드 (graceful reload)

        Steps:
            1. 새 워커 생성
            2. 새 모델 로드 + warm-up
            3. 트래픽 전환 (old → new)
            4. Old 워커 배출 + 정지

        Returns:
            리로드 성공 여부
        """
        logger.info("=== 모델 리로드 시작 ===")

        # 1. 새 워커 생성
        with self.lock:
            new_worker_id = f"worker-{self.worker_counter}"
            self.worker_counter += 1

        new_worker = ModelWorker(
            worker_id=new_worker_id,
            model_loader=self.model_loader,
            warmup_func=self.warmup_func,
            warmup_steps=self.warmup_steps
        )

        # 2. 새 모델 로드 + warm-up
        try:
            logger.info(f"[{new_worker_id}] 로드 + warm-up 시작")
            new_worker.initialize()
            logger.info(f"[{new_worker_id}] 로드 + warm-up 완료")

        except Exception as e:
            logger.error(f"[{new_worker_id}] 초기화 실패: {e}")
            return False

        # 3. 트래픽 전환
        with self.lock:
            old_worker = self.current_worker
            self.current_worker = new_worker

            logger.info(f"트래픽 전환: {old_worker.worker_id if old_worker else 'None'} → {new_worker_id}")

        # 4. Old 워커 배출 + 정지
        if old_worker:
            logger.info(f"[{old_worker.worker_id}] 배출 시작")
            old_worker.drain(timeout=self.drain_timeout)

            logger.info(f"[{old_worker.worker_id}] 정지")
            old_worker.stop()

        logger.info("=== 모델 리로드 완료 ===")
        return True

    def serve(self, request_func: Callable[[Any], Any], *args, **kwargs) -> Any:
        """
        요청 처리 (현재 워커 사용)

        Args:
            request_func: 요청 처리 함수
            *args, **kwargs: 요청 함수 인자

        Returns:
            요청 결과
        """
        if not self.current_worker:
            raise RuntimeError("워커가 초기화되지 않음")

        return self.current_worker.serve(request_func, *args, **kwargs)

    def get_current_worker_info(self) -> Optional[WorkerInfo]:
        """현재 워커 정보 조회"""
        if not self.current_worker:
            return None
        return self.current_worker.get_info()

    def shutdown(self):
        """매니저 종료"""
        logger.info("핫 리로드 매니저 종료")

        if self.current_worker:
            self.current_worker.drain(timeout=self.drain_timeout)
            self.current_worker.stop()
            self.current_worker = None


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 더미 모델
    class DummyModel:
        def __init__(self, version: int = 1):
            self.version = version
            logger.info(f"DummyModel v{version} 로드됨")

        def predict(self, x: int) -> int:
            return x * self.version

    # 모델 로더
    model_version = 1

    def load_model():
        return DummyModel(version=model_version)

    # Warm-up 함수
    def warmup(model):
        _ = model.predict(42)

    # 요청 처리 함수
    def process_request(model, x: int) -> int:
        return model.predict(x)

    # 핫 리로드 매니저 초기화
    manager = HotReloadManager(
        model_loader=load_model,
        warmup_func=warmup,
        warmup_steps=3,
        drain_timeout=5.0
    )

    # 첫 번째 워커 초기화
    print("\n=== 첫 번째 워커 초기화 ===\n")
    manager.initialize_first_worker()

    # 몇 가지 요청 처리
    print("\n=== 요청 처리 (v1) ===\n")
    for i in range(3):
        result = manager.serve(process_request, x=10)
        print(f"  Request {i+1}: 10 * v1 = {result}")

    # 워커 정보 조회
    print("\n=== 워커 정보 (v1) ===")
    info = manager.get_current_worker_info()
    if info:
        print(f"  Worker ID: {info.worker_id}")
        print(f"  Status: {info.status.value}")
        print(f"  Total Requests: {info.total_requests_served}")

    # 모델 업데이트 (v2)
    print("\n=== 모델 업데이트 (v1 → v2) ===\n")
    model_version = 2
    manager.reload()

    # 새 모델로 요청 처리
    print("\n=== 요청 처리 (v2) ===\n")
    for i in range(3):
        result = manager.serve(process_request, x=10)
        print(f"  Request {i+1}: 10 * v2 = {result}")

    # 워커 정보 조회
    print("\n=== 워커 정보 (v2) ===")
    info = manager.get_current_worker_info()
    if info:
        print(f"  Worker ID: {info.worker_id}")
        print(f"  Status: {info.status.value}")
        print(f"  Total Requests: {info.total_requests_served}")

    # 종료
    print("\n=== 매니저 종료 ===\n")
    manager.shutdown()

    print("테스트 완료")
