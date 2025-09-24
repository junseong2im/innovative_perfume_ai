"""
훈련 데이터셋 저장소

훈련 데이터셋 관리에 특화된 쿼리와 비즈니스 로직을 제공합니다.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from .base import BaseRepository
from ..database.models import TrainingDataset
from ..core.production_logging import get_logger

logger = get_logger(__name__)


class TrainingDatasetRepository(BaseRepository[TrainingDataset]):
    """훈련 데이터셋 저장소"""

    def __init__(self, session: Session):
        super().__init__(TrainingDataset, session)

    def find_by_type(self, dataset_type: str) -> List[TrainingDataset]:
        """데이터셋 타입별 조회"""
        return self.find_by(dataset_type=dataset_type)

    def find_by_status(self, status: str) -> List[TrainingDataset]:
        """상태별 조회"""
        return self.find_by(status=status)

    def get_ready_datasets(self) -> List[TrainingDataset]:
        """훈련 준비된 데이터셋 조회"""
        return self.find_by(status="ready")

    def get_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계"""
        try:
            total_datasets = self.count()

            # 타입별 분포
            type_distribution = {}
            results = self.session.query(
                TrainingDataset.dataset_type,
                func.count(TrainingDataset.id)
            ).group_by(TrainingDataset.dataset_type).all()

            for dataset_type, count in results:
                if dataset_type:
                    type_distribution[dataset_type] = count

            # 상태별 분포
            status_distribution = {}
            results = self.session.query(
                TrainingDataset.status,
                func.count(TrainingDataset.id)
            ).group_by(TrainingDataset.status).all()

            for status, count in results:
                if status:
                    status_distribution[status] = count

            # 총 샘플 수
            total_samples = self.session.query(
                func.sum(TrainingDataset.total_samples)
            ).scalar() or 0

            return {
                'total_datasets': total_datasets,
                'total_samples': total_samples,
                'type_distribution': type_distribution,
                'status_distribution': status_distribution
            }

        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {str(e)}")
            raise