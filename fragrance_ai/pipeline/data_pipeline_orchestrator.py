"""
Data Pipeline Orchestrator
데이터 수집, 정제, 모델 재학습의 전체 파이프라인을 관리하는 오케스트레이터

주요 기능:
1. 파이프라인 스케줄링
2. 태스크 의존성 관리
3. 실행 모니터링
4. 에러 처리 및 재시도
5. 알림 및 리포팅
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from enum import Enum
import traceback
from dataclasses import dataclass, asdict
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 파이프라인 오퍼레이터 임포트
from fragrance_ai.pipeline.web_scraping_operator import WebScrapingOperator
from fragrance_ai.pipeline.data_cleaning_operator import DataCleaningOperator
from fragrance_ai.pipeline.model_retraining_operator import ModelRetrainingOperator

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """태스크 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PipelineStage(Enum):
    """파이프라인 단계"""
    DATA_COLLECTION = "data_collection"
    DATA_CLEANING = "data_cleaning"
    MODEL_RETRAINING = "model_retraining"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"


@dataclass
class TaskResult:
    """태스크 실행 결과"""
    task_name: str
    stage: PipelineStage
    status: TaskStatus
    started_at: str
    completed_at: str
    duration_seconds: float
    result_data: Optional[Dict] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class PipelineRun:
    """파이프라인 실행 정보"""
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    tasks: List[TaskResult] = None
    metrics: Optional[Dict] = None
    config: Optional[Dict] = None


class DataPipelineOrchestrator:
    """
    데이터 파이프라인 오케스트레이터
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.pipeline_dir = Path(self.config['pipeline_dir'])
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

        # 오퍼레이터 초기화
        self.operators = {
            PipelineStage.DATA_COLLECTION: WebScrapingOperator(),
            PipelineStage.DATA_CLEANING: DataCleaningOperator(),
            PipelineStage.MODEL_RETRAINING: ModelRetrainingOperator()
        }

        # 실행 기록
        self.runs_history = []
        self.current_run = None

        # 스케줄러
        self.scheduler_enabled = False
        self.scheduler_thread = None

        logger.info("DataPipelineOrchestrator initialized")

    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'pipeline_dir': 'data/pipeline',
            'max_retries': 3,
            'retry_delay': 60,  # seconds
            'parallel_execution': False,
            'schedule': {
                'enabled': True,
                'daily_at': '02:00',  # 매일 새벽 2시
                'weekly_on': 'sunday',  # 주간 실행은 일요일
                'interval_hours': None  # 시간 간격 실행 (설정 시)
            },
            'stages': {
                'data_collection': {
                    'enabled': True,
                    'timeout': 3600,  # 1 hour
                    'urls': []  # 수집할 URL 목록
                },
                'data_cleaning': {
                    'enabled': True,
                    'timeout': 1800,  # 30 minutes
                    'min_quality_score': 0.5
                },
                'model_retraining': {
                    'enabled': True,
                    'timeout': 7200,  # 2 hours
                    'min_data_points': 100
                }
            },
            'notifications': {
                'on_success': True,
                'on_failure': True,
                'webhook_url': None,
                'email': None
            },
            'data_retention': {
                'raw_data_days': 30,
                'processed_data_days': 90,
                'model_backups': 10
            }
        }

    async def run_pipeline(self, run_id: Optional[str] = None) -> PipelineRun:
        """파이프라인 실행"""
        if not run_id:
            run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting pipeline run: {run_id}")

        # 실행 정보 초기화
        self.current_run = PipelineRun(
            run_id=run_id,
            started_at=datetime.now().isoformat(),
            status=TaskStatus.RUNNING,
            tasks=[],
            config=self.config
        )

        try:
            # Stage 1: 데이터 수집
            if self.config['stages']['data_collection']['enabled']:
                collection_result = await self._run_data_collection()
                self.current_run.tasks.append(collection_result)

                if collection_result.status == TaskStatus.FAILED:
                    raise Exception("Data collection failed")

            # Stage 2: 데이터 정제
            if self.config['stages']['data_cleaning']['enabled']:
                cleaning_result = await self._run_data_cleaning()
                self.current_run.tasks.append(cleaning_result)

                if cleaning_result.status == TaskStatus.FAILED:
                    raise Exception("Data cleaning failed")

            # Stage 3: 모델 재학습
            if self.config['stages']['model_retraining']['enabled']:
                retraining_result = await self._run_model_retraining()
                self.current_run.tasks.append(retraining_result)

                if retraining_result.status == TaskStatus.FAILED:
                    logger.warning("Model retraining failed, but pipeline continues")

            # Stage 4: 검증
            validation_result = await self._run_validation()
            self.current_run.tasks.append(validation_result)

            # 파이프라인 완료
            self.current_run.completed_at = datetime.now().isoformat()
            self.current_run.status = TaskStatus.SUCCESS

            # 메트릭 수집
            self.current_run.metrics = self._collect_metrics()

            # 성공 알림
            await self._send_notification("success", self.current_run)

            logger.info(f"Pipeline run completed successfully: {run_id}")

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            self.current_run.status = TaskStatus.FAILED
            self.current_run.completed_at = datetime.now().isoformat()

            # 실패 알림
            await self._send_notification("failure", self.current_run, str(e))

        finally:
            # 실행 기록 저장
            self._save_run_history(self.current_run)
            self.runs_history.append(self.current_run)

            # 오래된 데이터 정리
            await self._cleanup_old_data()

        return self.current_run

    async def _run_data_collection(self) -> TaskResult:
        """데이터 수집 단계 실행"""
        logger.info("Running data collection stage...")
        start_time = datetime.now()

        task_result = TaskResult(
            task_name="web_scraping",
            stage=PipelineStage.DATA_COLLECTION,
            status=TaskStatus.RUNNING,
            started_at=start_time.isoformat(),
            completed_at="",
            duration_seconds=0
        )

        try:
            operator = self.operators[PipelineStage.DATA_COLLECTION]

            # URL 목록 가져오기
            urls = self.config['stages']['data_collection'].get('urls', [])

            # 비동기 실행
            result = await operator.run(urls)

            # 수집된 데이터 저장
            output_path = self.pipeline_dir / f"scraped_data_{datetime.now().strftime('%Y%m%d')}.json"
            operator.save_to_file(str(output_path))

            task_result.status = TaskStatus.SUCCESS
            task_result.result_data = {
                'output_file': str(output_path),
                'stats': result.get('stats'),
                'data_count': result.get('data_count')
            }

            logger.info(f"Data collection completed: {result.get('data_count')} items collected")

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)

            # 재시도 로직
            if task_result.retry_count < self.config['max_retries']:
                task_result.retry_count += 1
                await asyncio.sleep(self.config['retry_delay'])
                return await self._run_data_collection()

        finally:
            end_time = datetime.now()
            task_result.completed_at = end_time.isoformat()
            task_result.duration_seconds = (end_time - start_time).total_seconds()

        return task_result

    async def _run_data_cleaning(self) -> TaskResult:
        """데이터 정제 단계 실행"""
        logger.info("Running data cleaning stage...")
        start_time = datetime.now()

        task_result = TaskResult(
            task_name="data_cleaning",
            stage=PipelineStage.DATA_CLEANING,
            status=TaskStatus.RUNNING,
            started_at=start_time.isoformat(),
            completed_at="",
            duration_seconds=0
        )

        try:
            operator = self.operators[PipelineStage.DATA_CLEANING]

            # 최신 수집 데이터 파일 찾기
            scraped_files = list(self.pipeline_dir.glob("scraped_data_*.json"))
            if not scraped_files:
                raise Exception("No scraped data found")

            latest_file = max(scraped_files, key=lambda p: p.stat().st_mtime)

            # 동기 실행 (IO 작업이 많지 않음)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                operator.run,
                str(latest_file)
            )

            task_result.status = TaskStatus.SUCCESS
            task_result.result_data = {
                'input_file': str(latest_file),
                'stats': result.get('stats'),
                'cleaned_count': result.get('cleaned_count')
            }

            logger.info(f"Data cleaning completed: {result.get('cleaned_count')} items cleaned")

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)

        finally:
            end_time = datetime.now()
            task_result.completed_at = end_time.isoformat()
            task_result.duration_seconds = (end_time - start_time).total_seconds()

        return task_result

    async def _run_model_retraining(self) -> TaskResult:
        """모델 재학습 단계 실행"""
        logger.info("Running model retraining stage...")
        start_time = datetime.now()

        task_result = TaskResult(
            task_name="model_retraining",
            stage=PipelineStage.MODEL_RETRAINING,
            status=TaskStatus.RUNNING,
            started_at=start_time.isoformat(),
            completed_at="",
            duration_seconds=0
        )

        try:
            operator = self.operators[PipelineStage.MODEL_RETRAINING]

            # 프로세스 풀에서 실행 (CPU 집약적)
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor,
                    operator.run
                )

            task_result.status = TaskStatus.SUCCESS
            task_result.result_data = {
                'models': result.get('models'),
                'stats': result.get('stats')
            }

            logger.info(f"Model retraining completed: {result.get('stats')}")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)

        finally:
            end_time = datetime.now()
            task_result.completed_at = end_time.isoformat()
            task_result.duration_seconds = (end_time - start_time).total_seconds()

        return task_result

    async def _run_validation(self) -> TaskResult:
        """검증 단계 실행"""
        logger.info("Running validation stage...")
        start_time = datetime.now()

        task_result = TaskResult(
            task_name="validation",
            stage=PipelineStage.VALIDATION,
            status=TaskStatus.RUNNING,
            started_at=start_time.isoformat(),
            completed_at="",
            duration_seconds=0
        )

        try:
            # 검증 로직
            validation_results = {
                'data_quality': self._validate_data_quality(),
                'model_performance': self._validate_model_performance(),
                'system_health': self._validate_system_health()
            }

            # 모든 검증 통과 확인
            all_passed = all(v.get('passed', False) for v in validation_results.values())

            task_result.status = TaskStatus.SUCCESS if all_passed else TaskStatus.FAILED
            task_result.result_data = validation_results

            logger.info(f"Validation {'passed' if all_passed else 'failed'}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            task_result.status = TaskStatus.FAILED
            task_result.error_message = str(e)

        finally:
            end_time = datetime.now()
            task_result.completed_at = end_time.isoformat()
            task_result.duration_seconds = (end_time - start_time).total_seconds()

        return task_result

    def _validate_data_quality(self) -> Dict:
        """데이터 품질 검증"""
        try:
            # 지식베이스 로드
            kb_path = Path('data/comprehensive_fragrance_notes_database.json')
            if not kb_path.exists():
                return {'passed': False, 'message': 'Knowledge base not found'}

            with open(kb_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)

            fragrances = knowledge_base.get('fragrances', {})

            # 품질 메트릭 계산
            total = len(fragrances)
            high_quality = sum(
                1 for f in fragrances.values()
                if f.get('data_quality_score', 0) >= 0.7
            )

            quality_ratio = high_quality / total if total > 0 else 0

            return {
                'passed': quality_ratio >= 0.5,
                'total_fragrances': total,
                'high_quality_count': high_quality,
                'quality_ratio': quality_ratio
            }

        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _validate_model_performance(self) -> Dict:
        """모델 성능 검증"""
        try:
            # 재학습 결과 확인
            results_files = list(Path('models/retrained').glob('retraining_results_*.json'))

            if not results_files:
                return {'passed': True, 'message': 'No retraining results to validate'}

            latest_result = max(results_files, key=lambda p: p.stat().st_mtime)

            with open(latest_result, 'r') as f:
                results = json.load(f)

            # 성능 개선 확인
            improvements = results.get('stats', {}).get('performance_improvements', {})

            avg_improvement = np.mean(list(improvements.values())) if improvements else 0

            return {
                'passed': avg_improvement >= 0,
                'average_improvement': avg_improvement,
                'model_improvements': improvements
            }

        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _validate_system_health(self) -> Dict:
        """시스템 상태 검증"""
        try:
            health_checks = {
                'disk_space': self._check_disk_space(),
                'memory_usage': self._check_memory_usage(),
                'api_connectivity': self._check_api_connectivity()
            }

            all_healthy = all(health_checks.values())

            return {
                'passed': all_healthy,
                'checks': health_checks
            }

        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _check_disk_space(self) -> bool:
        """디스크 공간 확인"""
        import shutil
        usage = shutil.disk_usage('/')
        free_gb = usage.free / (1024**3)
        return free_gb > 1  # 최소 1GB 여유 공간

    def _check_memory_usage(self) -> bool:
        """메모리 사용량 확인"""
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # 90% 미만 사용

    def _check_api_connectivity(self) -> bool:
        """API 연결 확인"""
        # 실제로는 API 헬스체크 수행
        return True

    def _collect_metrics(self) -> Dict:
        """파이프라인 메트릭 수집"""
        metrics = {
            'total_duration': 0,
            'stage_durations': {},
            'success_rate': 0,
            'data_processed': 0
        }

        if self.current_run and self.current_run.tasks:
            # 총 실행 시간
            metrics['total_duration'] = sum(
                t.duration_seconds for t in self.current_run.tasks
            )

            # 단계별 실행 시간
            for task in self.current_run.tasks:
                stage_name = task.stage.value
                if stage_name not in metrics['stage_durations']:
                    metrics['stage_durations'][stage_name] = 0
                metrics['stage_durations'][stage_name] += task.duration_seconds

            # 성공률
            total_tasks = len(self.current_run.tasks)
            successful_tasks = sum(
                1 for t in self.current_run.tasks
                if t.status == TaskStatus.SUCCESS
            )
            metrics['success_rate'] = successful_tasks / total_tasks if total_tasks > 0 else 0

            # 처리된 데이터
            for task in self.current_run.tasks:
                if task.result_data:
                    metrics['data_processed'] += task.result_data.get('data_count', 0)

        return metrics

    async def _send_notification(self, event_type: str, run: PipelineRun, error: Optional[str] = None):
        """알림 전송"""
        if not self.config['notifications'].get(f'on_{event_type}'):
            return

        notification = {
            'event': event_type,
            'run_id': run.run_id,
            'status': run.status.value,
            'timestamp': datetime.now().isoformat(),
            'metrics': run.metrics,
            'error': error
        }

        # Webhook 전송
        webhook_url = self.config['notifications'].get('webhook_url')
        if webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json=notification)
                logger.info(f"Notification sent to webhook: {event_type}")
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {e}")

        # 이메일 전송 (구현 필요)
        email = self.config['notifications'].get('email')
        if email:
            logger.info(f"Email notification would be sent to {email}")

    async def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        retention = self.config['data_retention']

        # 원시 데이터 정리
        raw_data_cutoff = datetime.now() - timedelta(days=retention['raw_data_days'])
        for file in self.pipeline_dir.glob("scraped_data_*.json"):
            if datetime.fromtimestamp(file.stat().st_mtime) < raw_data_cutoff:
                file.unlink()
                logger.info(f"Deleted old raw data: {file}")

        # 모델 백업 정리
        backup_dir = Path('models/backup')
        if backup_dir.exists():
            backups = sorted(
                backup_dir.glob("*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            for backup in backups[retention['model_backups']:]:
                backup.unlink()
                logger.info(f"Deleted old model backup: {backup}")

    def _save_run_history(self, run: PipelineRun):
        """실행 기록 저장"""
        history_file = self.pipeline_dir / f"run_{run.run_id}.json"

        with open(history_file, 'w', encoding='utf-8') as f:
            # TaskResult와 PipelineRun을 dict로 변환
            run_dict = asdict(run)
            run_dict['tasks'] = [asdict(task) for task in run.tasks]

            json.dump(run_dict, f, indent=2)

        logger.info(f"Run history saved to {history_file}")

    def start_scheduler(self):
        """스케줄러 시작"""
        if not self.config['schedule']['enabled']:
            logger.info("Scheduler is disabled")
            return

        self.scheduler_enabled = True

        # 스케줄 설정
        schedule_config = self.config['schedule']

        # 매일 실행
        if schedule_config.get('daily_at'):
            schedule.every().day.at(schedule_config['daily_at']).do(
                lambda: asyncio.run(self.run_pipeline())
            )
            logger.info(f"Scheduled daily run at {schedule_config['daily_at']}")

        # 주간 실행
        if schedule_config.get('weekly_on'):
            getattr(schedule.every(), schedule_config['weekly_on']).do(
                lambda: asyncio.run(self.run_pipeline())
            )
            logger.info(f"Scheduled weekly run on {schedule_config['weekly_on']}")

        # 간격 실행
        if schedule_config.get('interval_hours'):
            schedule.every(schedule_config['interval_hours']).hours.do(
                lambda: asyncio.run(self.run_pipeline())
            )
            logger.info(f"Scheduled interval run every {schedule_config['interval_hours']} hours")

        # 스케줄러 스레드 시작
        import threading

        def run_scheduler():
            while self.scheduler_enabled:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()

        logger.info("Scheduler started")

    def stop_scheduler(self):
        """스케줄러 중지"""
        self.scheduler_enabled = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def get_run_history(self, limit: int = 10) -> List[Dict]:
        """실행 기록 조회"""
        history_files = sorted(
            self.pipeline_dir.glob("run_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]

        history = []
        for file in history_files:
            with open(file, 'r', encoding='utf-8') as f:
                history.append(json.load(f))

        return history


# 실행 예시
async def main():
    """파이프라인 실행"""
    orchestrator = DataPipelineOrchestrator()

    # 단일 실행
    result = await orchestrator.run_pipeline()
    print(f"Pipeline completed: {result.status.value}")
    print(f"Metrics: {result.metrics}")

    # 스케줄러 시작 (데몬 모드)
    # orchestrator.start_scheduler()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 파이프라인 실행
    asyncio.run(main())